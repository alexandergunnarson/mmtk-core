use super::cm::LXRConcurrentTraceObjects;
use super::cm::LXRStopTheWorldProcessEdges;
use super::SurvivalRatioPredictorLocal;
use super::LXR;
use crate::plan::VectorQueue;
use crate::scheduler::gc_work::RootKind;
use crate::scheduler::gc_work::ScanObjects;
use crate::scheduler::gc_work::SlotOf;
use crate::util::address::CLDScanPolicy;
use crate::util::address::RefScanPolicy;
use crate::util::copy::CopySemantics;
use crate::util::copy::GCWorkerCopyContext;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::rc::*;
use crate::vm::slot::MemorySlice;
use crate::vm::slot::Slot;
use crate::LazySweepingJobsCounter;
use crate::{
    plan::immix::Pause,
    policy::{immix::block::Block, space::Space},
    scheduler::{gc_work::ProcessEdgesBase, GCWork, GCWorker, ProcessEdgesWork, WorkBucketStage},
    util::{metadata::side_metadata, object_forwarding, ObjectReference},
    vm::*,
    MMTK,
};
use atomic::Ordering;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

// RC death diagnostic instrumentation statics.
// Only used when the `lxr_rc_death_diag` feature is enabled.
#[cfg(feature = "lxr_rc_death_diag")]
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "lxr_rc_death_diag")]
static DEATH_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "lxr_rc_death_diag")]
const DEATH_LOG_LIMIT: usize = 500;
#[cfg(feature = "lxr_rc_death_diag")]
static GC_CYCLE_COUNT: AtomicUsize = AtomicUsize::new(0);

// ========================================================================= //
// Per-object RC tracing diagnostic (feature: lxr_rc_trace).
//
// Logs every RC operation (inc, dec, promotion scan, death, recursive-dec)
// for a configurable set of target object addresses.  This is the primary
// tool for diagnosing systematic RC undercount: after a death diagnostic run
// (lxr_rc_death_diag) identifies dying addresses, set those addresses via
// rc_trace_add_addr() or MMTK_RC_TRACE_ADDRS env var, rebuild, and rerun.
// The trace log shows every RC change for those objects, revealing which
// expected increment is missing.
//
// Usage:
//   1. Build with --features lxr_rc_death_diag to identify dying addresses
//   2. Set MMTK_RC_TRACE_ADDRS=0x...,0x...,... (comma-separated hex addrs)
//   3. Build with --features lxr_rc_trace (can combine with lxr_rc_death_diag)
//   4. Run bootstrap — trace log on stderr shows every RC op for those objects
//
// Each log line has the format:
//   [rc-trace gc=N OP] 0xADDR details...
// where OP is one of:
//   inc          — RC incremented (old_rc → old_rc+1)
//   inc-slot     — slot processing caused an increment (shows slot addr, edge kind)
//   inc-promo    — promotion scan found this as a child (shows parent addr)
//   inc-promo-direct — direct RC inc during promotion (already mature child)
//   dec          — RC decremented (old_rc → old_rc-1)
//   death        — object freed (RC reached 0)
//   rec-dec      — recursive dec from a dying parent (shows parent addr)
//   promote      — object promoted from nursery (RC 0→1)
// ========================================================================= //

#[cfg(feature = "lxr_rc_trace")]
use std::sync::atomic::AtomicUsize as RcTraceAtomicUsize;

/// Maximum number of simultaneously traced object addresses.
#[cfg(feature = "lxr_rc_trace")]
const RC_TRACE_MAX_ADDRS: usize = 16;

/// Target addresses for RC tracing.  Set via rc_trace_add_addr() or
/// parsed from MMTK_RC_TRACE_ADDRS env var during mmtk_gc_init.
#[cfg(feature = "lxr_rc_trace")]
static RC_TRACE_ADDRS: [RcTraceAtomicUsize; RC_TRACE_MAX_ADDRS] = {
    // const initializer — array of AtomicUsize::new(0)
    const ZERO: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);
    [ZERO; RC_TRACE_MAX_ADDRS]
};

/// Number of active trace addresses (monotonically increasing, capped at RC_TRACE_MAX_ADDRS).
#[cfg(feature = "lxr_rc_trace")]
static RC_TRACE_COUNT: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);

/// Optional VTAG to trace.  When non-zero, EVERY object whose Julia type tag
/// (the word at `obj-8`, with the low GC bits masked) equals this value is
/// traced — not just the explicit per-address set.  This makes type-targeted
/// tracing DETERMINISTIC across the nondeterministic bootstrap: e.g. set
/// `MMTK_RC_TRACE_VTAG=0x200ffc01c00` to trace all TypeMapEntry objects and
/// observe the full inc/promote/dec/death pattern of the undercounted class.
/// Set via `MMTK_RC_TRACE_VTAG` env var during `mmtk_gc_init`.
#[cfg(feature = "lxr_rc_trace")]
static RC_TRACE_VTAG: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);

/// GC cycle counter for trace log timestamps.
#[cfg(feature = "lxr_rc_trace")]
static RC_TRACE_GC_COUNT: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);

/// Counters for the corruption guard in `process_slot` (rate-limiting).
#[cfg(feature = "lxr_rc_trace")]
static CORRUPT_LOG_COUNT: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);
#[cfg(feature = "lxr_rc_trace")]
const CORRUPT_LOG_LIMIT: usize = 50;
#[cfg(feature = "lxr_rc_trace")]
const CORRUPT_SCAN_LIMIT: usize = 5;

// ===================================================================== //
// Death-time genuine-undercount detector (§2.17 step 1/2).
//
// The corruption guard in `process_slot`/`scan_nursery_object` only fires
// when a GC inc-slot or promotion path touches the dangling object.  But the
// §2.17 frontier is nondeterministic: the freed-and-reused young object is
// often consumed by the MUTATOR (a corrupted AST node read during lowering)
// BEFORE any GC scans it, so no guard fires and we get a plain SIGSEGV with
// no diagnostic.
//
// This detector closes that gap by catching the bug at the MOMENT OF DEATH:
// when an object is freed, scan the live heap for any external referrer.  A
// freed object with a LIVE heap referrer (matches>0) is the genuine RC
// undercount — the slot whose missing inc/barrier is the root-cause bug.
// (§2.16 proved the 501 BindingPartition deaths have matches=0 — they are
// benign; this detector specifically excludes those by reporting only
// matches>0.)
//
// The referrer scan is O(live-heap) per death, so it is BUDGETED:
//   - Only runs from GC cycle `MMTK_RC_UNDERCOUNT_FROM_GC` onward (the
//     undercount surfaces late, ~GC 27; default 0 = always).
//   - At most `MMTK_RC_UNDERCOUNT_BUDGET` scans total (default 200).
// When a death with a live referrer is found, it logs the dying object's
// vtag + every referrer word (verbose scan) so the storing slot/parent is
// identified, then `MMTK_RC_TRACE_ADDRS` can be pointed at it.
#[cfg(feature = "lxr_rc_trace")]
static UNDERCOUNT_SCAN_BUDGET: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);
#[cfg(feature = "lxr_rc_trace")]
static UNDERCOUNT_FROM_GC: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);
#[cfg(feature = "lxr_rc_trace")]
static UNDERCOUNT_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(feature = "lxr_rc_trace")]
static UNDERCOUNT_FOUND: RcTraceAtomicUsize = RcTraceAtomicUsize::new(0);
#[cfg(feature = "lxr_rc_trace")]
const UNDERCOUNT_FOUND_LIMIT: usize = 20;

/// Parse `MMTK_RC_UNDERCOUNT_BUDGET` / `MMTK_RC_UNDERCOUNT_FROM_GC` to enable
/// the death-time genuine-undercount detector.  Setting either enables it.
#[cfg(feature = "lxr_rc_trace")]
pub fn rc_undercount_init_from_env() {
    let mut enabled = false;
    if let Ok(v) = std::env::var("MMTK_RC_UNDERCOUNT_BUDGET") {
        if let Ok(n) = v.trim().parse::<usize>() {
            UNDERCOUNT_SCAN_BUDGET.store(n, Ordering::SeqCst);
            enabled = true;
        }
    } else {
        // Default budget if the detector is enabled only via FROM_GC.
        UNDERCOUNT_SCAN_BUDGET.store(200, Ordering::SeqCst);
    }
    if let Ok(v) = std::env::var("MMTK_RC_UNDERCOUNT_FROM_GC") {
        if let Ok(n) = v.trim().parse::<usize>() {
            UNDERCOUNT_FROM_GC.store(n, Ordering::SeqCst);
            enabled = true;
        }
    }
    if enabled {
        UNDERCOUNT_ENABLED.store(true, Ordering::SeqCst);
        eprintln!(
            "[rc-undercount] death-time genuine-undercount detector ENABLED (budget={}, from_gc={})",
            UNDERCOUNT_SCAN_BUDGET.load(Ordering::SeqCst),
            UNDERCOUNT_FROM_GC.load(Ordering::SeqCst),
        );
    }
}

/// Add an address to the RC trace set.  Thread-safe (uses atomic CAS on the count).
/// Returns true if the address was added, false if the set is full.
#[cfg(feature = "lxr_rc_trace")]
pub fn rc_trace_add_addr(addr: usize) {
    let idx = RC_TRACE_COUNT.fetch_add(1, Ordering::SeqCst);
    if idx < RC_TRACE_MAX_ADDRS {
        RC_TRACE_ADDRS[idx].store(addr, Ordering::SeqCst);
        eprintln!("[rc-trace] Tracing address {:#x} (slot {})", addr, idx);
    } else {
        RC_TRACE_COUNT.fetch_sub(1, Ordering::SeqCst);
        eprintln!(
            "[rc-trace] WARNING: trace set full (max {}), ignoring {:#x}",
            RC_TRACE_MAX_ADDRS, addr
        );
    }
}

/// Initialize trace addresses from the MMTK_RC_TRACE_ADDRS environment variable.
/// Format: comma-separated hex addresses, e.g. "0x200ffc01000,0x200ffc02000"
#[cfg(feature = "lxr_rc_trace")]
pub fn rc_trace_init_from_env() {
    if let Ok(val) = std::env::var("MMTK_RC_TRACE_ADDRS") {
        for part in val.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let addr_str = part
                .strip_prefix("0x")
                .or_else(|| part.strip_prefix("0X"))
                .unwrap_or(part);
            match usize::from_str_radix(addr_str, 16) {
                Ok(addr) => rc_trace_add_addr(addr),
                Err(e) => eprintln!("[rc-trace] WARNING: failed to parse '{}': {}", part, e),
            }
        }
    }
    // Optional deterministic type-targeted tracing by vtag.
    if let Ok(val) = std::env::var("MMTK_RC_TRACE_VTAG") {
        let s = val.trim();
        let hex = s
            .strip_prefix("0x")
            .or_else(|| s.strip_prefix("0X"))
            .unwrap_or(s);
        match usize::from_str_radix(hex, 16) {
            Ok(vtag) => {
                RC_TRACE_VTAG.store(vtag & !0xfusize, Ordering::SeqCst);
                eprintln!(
                    "[rc-trace] Tracing ALL objects with vtag {:#x}",
                    vtag & !0xfusize
                );
            }
            Err(e) => eprintln!(
                "[rc-trace] WARNING: failed to parse MMTK_RC_TRACE_VTAG '{}': {}",
                val, e
            ),
        }
    }
}

/// Increment the trace GC cycle counter.  Call once per GC cycle.
#[cfg(feature = "lxr_rc_trace")]
pub fn rc_trace_inc_gc_count() {
    RC_TRACE_GC_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Check whether an object is in the RC trace set.
#[cfg(feature = "lxr_rc_trace")]
#[inline]
pub(super) fn is_rc_traced(o: ObjectReference) -> bool {
    let addr = o.to_raw_address().as_usize();
    let count = RC_TRACE_COUNT.load(Ordering::Relaxed);
    for i in 0..count.min(RC_TRACE_MAX_ADDRS) {
        if RC_TRACE_ADDRS[i].load(Ordering::Relaxed) == addr {
            return true;
        }
    }
    // VTAG match (deterministic type-targeted tracing).  Read the tag word at
    // `obj-8` and mask the low 4 GC/marking bits that Julia stores in the tag.
    let want_vtag = RC_TRACE_VTAG.load(Ordering::Relaxed);
    if want_vtag != 0 {
        let tag_addr = o.to_raw_address() - 8usize;
        if tag_addr.is_mapped() {
            let raw = unsafe { tag_addr.load::<usize>() };
            if (raw & !0xfusize) == want_vtag {
                return true;
            }
        }
    }
    false
}

/// Get current GC cycle for trace log.
#[cfg(feature = "lxr_rc_trace")]
#[inline]
pub(super) fn rc_trace_gc() -> usize {
    RC_TRACE_GC_COUNT.load(Ordering::Relaxed)
}

/// Diagnostic (lxr_rc_trace): walk the entire live heap (Immix + LOS) and
/// report every word in any live block that holds the address of `target`.
/// Used to find the "untracked referrer" of a dying/corrupt object.
///
/// This performs a RAW word-by-word memory scan of every allocated block (not a
/// VO-bit object walk), because under LXR the VO bits are not a reliable object
/// map during the lazy RC sweep.  If this reports ZERO matches, the dangling
/// pointer is a C stack local (a missing GC root), not a missing barrier.
#[cfg(feature = "lxr_rc_trace")]
pub(super) fn scan_heap_referrers<VM: VMBinding>(target: ObjectReference, lxr: &LXR<VM>) {
    let (words, matches) = scan_heap_referrers_collect(target, lxr, true);
    eprintln!(
        "[rc-trace gc={} referrer-scan-done] target={:#x} words_scanned={} matches={}",
        rc_trace_gc(),
        target.to_raw_address().as_usize(),
        words,
        matches,
    );
}

/// Diagnostic (lxr_rc_trace): given the address of a heap WORD that holds a
/// pointer to a freed object (a "referrer word" / dangling slot), walk
/// backward word-by-word to find the OWNING Julia object — i.e. the object
/// whose field that word is.  Returns a human-readable description of the
/// owner (its type name + the field byte-offset of the dangling slot) so the
/// type/store-site with the missing RC inc/barrier can be identified.
///
/// VO bits are unreliable mid-sweep (see `scan_heap_referrers`), so this does
/// NOT use `find_object_from_internal_pointer`.  Instead it scans backward up
/// to `MAX_BACK` bytes for the first candidate object-start whose tag-at-−8 is
/// a valid DataType (`debug_object_tag_is_valid`) AND whose size covers the
/// referrer word.  Bounded and best-effort: returns `<owner-not-found>` if no
/// plausible owner is located within the search window.
#[cfg(feature = "lxr_rc_trace")]
pub(super) fn describe_referrer_owner<VM: VMBinding>(slot_addr: usize) -> String {
    describe_referrer_owner_with_lxr::<VM>(slot_addr, None)
}

/// Decisive liveness filter for the genuine-undercount detector.
///
/// `enumerate_objects` walks every word of every ALLOCATED block, including
/// dead-but-not-yet-swept objects and stale data in free holes (LXR sweeps
/// lazily).  So a raw word-match against a freed target is NOT proof of an RC
/// undercount: the matching word may itself live in DEAD memory (a dangling
/// pointer in an object that is also about to be reclaimed).
///
/// This walks backward from the referrer word to the nearest plausible object
/// start (valid tag-at-−8) and returns `true` only if that owner is GENUINELY
/// LIVE — i.e. `rc>0` OR `marked` (in immix/los), or it lives in a
/// non-RC-tracked permanent space (VM/immortal).  An owner that is `rc==0` and
/// `!marked` is dead-unswept, so the referrer is a FALSE POSITIVE and must not
/// be counted as an undercount.
#[cfg(feature = "lxr_rc_trace")]
pub(super) fn referrer_owner_is_live<VM: VMBinding>(slot_addr: usize, lxr: &LXR<VM>) -> bool {
    use crate::policy::space::Space;
    use crate::util::Address;
    const MAX_BACK: usize = 4096;
    let word = std::mem::size_of::<usize>();
    let mut cand = (slot_addr & !(word - 1)).saturating_sub(word);
    let lo = slot_addr.saturating_sub(MAX_BACK);
    while cand >= lo {
        let cand_addr = unsafe { Address::from_usize(cand) };
        if cand_addr.is_mapped() && (cand_addr - 8usize).is_mapped() {
            let oref = unsafe { ObjectReference::from_raw_address_unchecked(cand_addr) };
            if VM::VMScanning::debug_object_tag_is_valid(oref) {
                let in_immix = lxr.immix_space.in_space(oref);
                let in_los = lxr.los().in_space(oref);
                if in_immix || in_los {
                    let live = RefCountHelper::<VM>::NEW.count(oref) > 0 || lxr.is_marked(oref);
                    // Found the nearest RC-tracked owner: its liveness is
                    // authoritative.  (If it is dead, keep searching backward
                    // only if a closer dead object could shadow a live one —
                    // but the NEAREST valid owner is the real container, so we
                    // return its verdict directly.)
                    return live;
                } else {
                    // Non-RC-tracked permanent space (VM/immortal/nonmoving):
                    // always live.
                    return true;
                }
            }
        }
        if cand < word {
            break;
        }
        cand -= word;
    }
    // No plausible owner found within the window — conservatively treat as NOT
    // live (a dangling word in a free hole has no object owner).
    false
}

#[cfg(feature = "lxr_rc_trace")]
pub(super) fn describe_referrer_owner_with_lxr<VM: VMBinding>(
    slot_addr: usize,
    lxr: Option<&LXR<VM>>,
) -> String {
    use crate::util::Address;
    // Object headers are 16-byte aligned in Julia's allocator; the tag is at
    // obj-8, data starts at obj.  Search backward from the slot's own word.
    //
    // The heap is dense, so several backward addresses may have a `-8` word
    // that spuriously validates as a DataType.  To disambiguate, we collect up
    // to a few candidate owners (valid tag-at-−8 + size covers the slot) and
    // report all of them; the true owner is the one whose type + field offset
    // matches a known pointer field of that type.  We exclude `cand ==
    // slot_addr` (field_off 0 where the slot's OWN location is treated as an
    // object header — a common spurious self-match), since a genuine
    // pointer-bearing object almost never starts exactly at the dangling slot.
    //
    // SAFETY: this runs while the heap is mid-sweep, so it uses ONLY
    // single-word reads guarded by `is_mapped()` and `debug_object_type_name`
    // (which does not traverse the type layout or any data fields).  It does
    // NOT call `get_size`/`debug_describe_object`, which can fault on a
    // partially-freed object.  Because exact size is unavailable, it reports
    // EVERY backward candidate (up to MAX_CANDS) whose tag-at-−8 is a valid
    // DataType and whose distance to the slot is within a conservative max
    // object span; the true owner is the nearest candidate whose type's
    // pointer-field layout includes `field_off`.
    const MAX_BACK: usize = 4096;
    const MAX_CANDS: usize = 4;
    let word = std::mem::size_of::<usize>();
    let mut cand = (slot_addr & !(word - 1)).saturating_sub(word);
    let lo = slot_addr.saturating_sub(MAX_BACK);
    let mut out = String::new();
    let mut found = 0usize;
    while cand >= lo && found < MAX_CANDS {
        let cand_addr = unsafe { Address::from_usize(cand) };
        if cand_addr.is_mapped() && (cand_addr - 8usize).is_mapped() {
            let oref = unsafe { ObjectReference::from_raw_address_unchecked(cand_addr) };
            if VM::VMScanning::debug_object_tag_is_valid(oref) {
                let tn = VM::VMScanning::debug_object_type_name(oref);
                if !tn.is_empty() {
                    let field_off = slot_addr - cand;
                    if !out.is_empty() {
                        out.push_str(" | ");
                    }
                    // Report the owner candidate's RC count, space, and mark
                    // state.  This is decisive for Class B: if the true owner
                    // (a mature cache buffer) has rc>0 and lives in immix, then
                    // its field unlog bits should be UNLOGGED and the store of
                    // the dying object SHOULD have fired the barrier.  If the
                    // owner has rc==0 (untracked) or is unmarked, the buffer
                    // itself was never RC-tracked → its slots never re-armed.
                    let (rc_str, space_str, marked_str) = if let Some(lxr) = lxr {
                        let rc = RefCountHelper::<VM>::NEW.count(oref);
                        let sp = if lxr.immix_space.in_space(oref) {
                            "immix"
                        } else if lxr.los().in_space(oref) {
                            "los"
                        } else {
                            "other"
                        };
                        (format!("{}", rc), sp, format!("{}", lxr.is_marked(oref)))
                    } else {
                        ("?".to_string(), "?", "?".to_string())
                    };
                    out.push_str(&format!(
                        "cand_owner={:#x} field_off={} type={} owner_rc={} owner_space={} owner_marked={}",
                        cand, field_off, tn, rc_str, space_str, marked_str,
                    ));
                    found += 1;
                }
            }
        }
        if cand < word {
            break;
        }
        cand -= word;
    }
    if out.is_empty() {
        "<owner-not-found>".to_string()
    } else {
        out
    }
}

/// Diagnostic (lxr_rc_trace): name the space that a referrer WORD lives in.
/// Helps classify whether a dangling pointer is held by a mature Immix object,
/// a LOS buffer, the VM space (sysimage), the immortal space, or the nonmoving
/// space — which narrows down the missing-barrier store site.
#[cfg(feature = "lxr_rc_trace")]
pub(super) fn referrer_space_name<VM: VMBinding>(
    a: crate::util::Address,
    lxr: &LXR<VM>,
) -> &'static str {
    use crate::policy::space::Space;
    if lxr.immix_space.address_in_space(a) {
        "immix"
    } else if lxr.los().address_in_space(a) {
        "los"
    } else if lxr.common.immortal.address_in_space(a) {
        "immortal"
    } else {
        "other"
    }
}

/// Diagnostic (lxr_rc_trace): the core raw heap-wide reverse-reference scan.
/// Walks every word of every live Immix + LOS block and counts (optionally
/// logging) every word that equals `target`'s address but lives OUTSIDE
/// `target`'s own page (so self-referential `next`/`restriction` fields are not
/// counted as external referrers).  Returns `(words_scanned, matches)`.
///
/// `verbose=true` logs each matching word's address; `verbose=false` is the
/// quiet variant used by the death-time genuine-undercount detector, which
/// scans MANY deaths and only wants the match count to decide whether a death
/// is a real RC undercount (a freed object still pointed to by a live slot).
#[cfg(feature = "lxr_rc_trace")]
pub(super) fn scan_heap_referrers_collect<VM: VMBinding>(
    target: ObjectReference,
    lxr: &LXR<VM>,
    verbose: bool,
) -> (usize, usize) {
    use crate::policy::space::Space;
    use crate::util::object_enum::ObjectEnumerator;
    use crate::util::Address;

    let target_addr = target.to_raw_address().as_usize();

    // Collect referrer-word addresses during the (single) heap walk WITHOUT
    // doing any re-entrant heap reads (owner backward-scan, hexdump, unlog-bit
    // reads).  Those secondary reads are deferred until AFTER enumeration
    // completes, because performing them inside `enumerate_objects` (which
    // holds the space's block iterator mid-sweep) can fault on a
    // partially-reclaimed neighbour.  This is why the previous "do a quiet
    // scan, then a second verbose scan" approach SIGSEGV'd on the second walk:
    // a single collect-then-describe pass is both cheaper and crash-safe.
    struct RawWordScanner {
        target: usize,
        self_page: usize,
        matches: usize,
        words: usize,
        collect: bool,
        hits: Vec<usize>,
    }
    impl ObjectEnumerator for RawWordScanner {
        fn visit_object(&mut self, _object: ObjectReference) {}
        fn visit_address_range(&mut self, start: Address, end: Address) {
            let mut a = start;
            while a < end {
                if a.is_mapped() {
                    let w = unsafe { a.load::<usize>() };
                    self.words += 1;
                    // Skip words inside the target's own page (self-references
                    // in its `next`/`restriction` are not external referrers).
                    if w == self.target && (a.as_usize() & !0xfff) != self.self_page {
                        if self.collect && self.hits.len() < 64 {
                            self.hits.push(a.as_usize());
                        }
                        self.matches += 1;
                    }
                }
                a += std::mem::size_of::<usize>();
            }
        }
    }

    let mut scanner = RawWordScanner {
        target: target_addr,
        self_page: target_addr & !0xfff,
        matches: 0,
        words: 0,
        collect: true, // always collect hits so we can apply the liveness filter
        hits: Vec::new(),
    };
    lxr.immix_space.enumerate_objects(&mut scanner);
    lxr.los().enumerate_objects(&mut scanner);

    // Decisive correction (§2.18 re-examination): `enumerate_objects` walks ALL
    // allocated memory, including dead-but-unswept objects and stale words in
    // free holes.  A raw word-match is therefore NOT a genuine undercount on
    // its own.  Re-count only hits whose OWNER object is genuinely live
    // (rc>0 || marked, or a permanent space).  `matches` returned to callers is
    // this LIVE count — the real undercount signal.
    let live_matches = scanner
        .hits
        .iter()
        .filter(|at| referrer_owner_is_live::<VM>(**at, lxr))
        .count();

    // Only emit the (expensive, verbose) per-referrer dump when there is at
    // least one GENUINELY LIVE referrer — i.e. a real undercount.  This keeps
    // the proven-benign dead-unswept matches from flooding the log.
    if verbose && live_matches > 0 {
        for at in &scanner.hits {
            let a = unsafe { Address::from_usize(*at) };
            let owner_live = referrer_owner_is_live::<VM>(*at, lxr);
            // Whether the slot's FIELD UNLOG BIT is currently "logged" (0).
            // A `logged` referrer slot means the LXR field-logging barrier
            // would SKIP a store into it, so the stored object was never
            // RC-incremented: that IS the missing-barrier mechanism for the
            // undercount.
            let logged = a.is_field_logged::<VM>();
            eprintln!(
                "[rc-trace referrer-word] at={:#x} space={} owner_live={} field_logged={} -> {:#x} {}",
                a.as_usize(),
                referrer_space_name::<VM>(a, lxr),
                owner_live,
                logged,
                target_addr,
                describe_referrer_owner_with_lxr::<VM>(a.as_usize(), Some(lxr)),
            );
            // Raw hexdump of the 16 words preceding (and incl.) the referrer
            // slot, to manually decode the owning object header (tag-at-−8)
            // when the backward-scan owner-finder is ambiguous in a dense heap.
            let mut dump = String::new();
            for k in (0..16isize).rev() {
                let wa = a - (k as usize) * std::mem::size_of::<usize>();
                if wa.is_mapped() {
                    dump.push_str(&format!(" [{:#x}]={:#x}", wa.as_usize(), unsafe {
                        wa.load::<usize>()
                    },));
                }
            }
            eprintln!("[rc-trace referrer-dump]{}", dump);
        }
    }
    // Return the LIVE referrer count (the genuine undercount signal), NOT the
    // raw word-match count `scanner.matches` (which includes dangling pointers
    // in dead-but-unswept memory and stale words in free holes).
    (scanner.words, live_matches)
}

/// Death-time genuine-undercount detector (§2.17).  Called from
/// `process_dead_object` for every freed object.  If the detector is enabled
/// (via `MMTK_RC_UNDERCOUNT_BUDGET`/`_FROM_GC`) and we are at/after the
/// configured GC cycle with budget remaining, do a quiet heap-wide
/// reverse-reference scan.  If the freed object STILL has a live external
/// referrer (matches>0), this is the genuine RC undercount: log the dying
/// object's vtag/size, then do a VERBOSE scan to print every referrer word so
/// the storing slot/parent is identified.  Returns immediately (near-zero
/// cost) when disabled, out of budget, or before the configured GC cycle.
#[cfg(feature = "lxr_rc_trace")]
#[cold]
pub(super) fn detect_death_undercount<VM: VMBinding>(o: ObjectReference, lxr: &LXR<VM>) {
    if !UNDERCOUNT_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    if rc_trace_gc() < UNDERCOUNT_FROM_GC.load(Ordering::Relaxed) {
        return;
    }
    // Spend one unit of budget (saturating-ish via CAS-free fetch_sub guard).
    let remaining = UNDERCOUNT_SCAN_BUDGET.load(Ordering::Relaxed);
    if remaining == 0 {
        return;
    }
    UNDERCOUNT_SCAN_BUDGET.fetch_sub(1, Ordering::Relaxed);

    // Single verbose collect-then-describe scan (see
    // `scan_heap_referrers_collect`): the previous code did one quiet scan
    // followed by a second verbose scan, and the second full heap walk
    // reproducibly SIGSEGV'd mid-sweep.  We now do exactly one walk.  If there
    // are no live referrers, the per-hit describe loop simply runs zero times.
    let (_words, matches) = scan_heap_referrers_collect(o, lxr, true);
    if matches == 0 {
        // Benign death (no live referrer) — e.g. the proven-benign 501
        // BindingPartition deaths (§2.16).  Do not report.
        return;
    }
    let found = UNDERCOUNT_FOUND.fetch_add(1, Ordering::Relaxed);
    if found >= UNDERCOUNT_FOUND_LIMIT {
        return;
    }
    let obj_addr = o.to_raw_address();
    let tag_addr = obj_addr - 8usize;
    let vtag: usize = if tag_addr.is_mapped() {
        unsafe { tag_addr.load::<usize>() }
    } else {
        0
    };
    let size = if obj_addr.is_mapped() {
        o.get_size::<VM>()
    } else {
        0
    };
    // Report the dying object's own mark state.  If it is MARKED (mark-live)
    // but freed with rc==0, that is the RC-vs-trace inconsistency: the
    // concurrent tracer reached it (so a live object points to it) yet RC
    // drove it to 0 and reclaimed it.
    let marked = lxr.is_marked(o);
    // Name the dying object's type from the (still-mapped) vtag rather than
    // `debug_describe_object`, which calls `get_size`/layout traversal and can
    // fault on a mid-free object.
    let desc = {
        let vt = unsafe { ObjectReference::from_raw_address_unchecked(o.to_raw_address()) };
        VM::VMScanning::debug_object_type_name(vt)
    };
    eprintln!(
        "[rc-undercount gc={} GENUINE-UNDERCOUNT #{}] freed={:#x} vtag={:#x} size={} live_referrers={} marked={} desc={} — a LIVE slot still references this FREED object (missing inc/barrier)",
        rc_trace_gc(),
        found,
        obj_addr.as_usize(),
        vtag,
        size,
        matches,
        marked,
        desc,
    );
}

/// Check whether a raw address (e.g. a slot address) is in the RC trace set.
#[cfg(feature = "lxr_rc_trace")]
#[inline]
pub(super) fn is_rc_traced_addr(addr: usize) -> bool {
    let count = RC_TRACE_COUNT.load(Ordering::Relaxed);
    for i in 0..count.min(RC_TRACE_MAX_ADDRS) {
        if RC_TRACE_ADDRS[i].load(Ordering::Relaxed) == addr {
            return true;
        }
    }
    false
}

/// Check whether an object resides in a space that has RC_TABLE side metadata
/// mapped (Immix or LOS).  Objects in other spaces (VM space, immortal, non-moving)
/// do not have RC metadata and must be skipped by all RC operations.
#[inline]
pub(super) fn has_rc_metadata<VM: VMBinding>(o: ObjectReference, lxr: &LXR<VM>) -> bool {
    // Positive filter: only Immix and LOS have RC_TABLE metadata.
    // With the fix to address_in_space() for discontiguous spaces, in_space()
    // is now precise (uses chunk-based descriptor lookup, not range check).
    lxr.immix_space.in_space(o) || lxr.los().in_space(o)
}

#[inline]
fn prefetch_object<VM: VMBinding>(o: ObjectReference, rc: &RefCountHelper<VM>) {
    if crate::args::PREFETCH_HEADER {
        o.prefetch_read();
    }
    if crate::args::PREFETCH_RC {
        rc.prefetch_read(o);
    }
}

pub struct ProcessIncs<VM: VMBinding, const KIND: EdgeKind> {
    /// Increments to process
    incs: Vec<VM::VMSlot>,
    inc_slices: Vec<VM::VMMemorySlice>,
    /// Recursively generated new increments
    new_incs: VectorQueue<VM::VMSlot>,
    new_inc_slices: VectorQueue<VM::VMMemorySlice>,
    new_incs_count: u32,
    pause: Pause,
    in_cm: bool,
    no_evac: bool,
    pub root_kind: Option<RootKind>,
    depth: u32,
    lxr: &'static LXR<VM>,
    rc: RefCountHelper<VM>,
    survival_ratio_predictor_local: SurvivalRatioPredictorLocal,
    copy_context: *mut GCWorkerCopyContext<VM>,
}

unsafe impl<VM: VMBinding, const KIND: EdgeKind> Send for ProcessIncs<VM, KIND> {}

impl<VM: VMBinding, const KIND: EdgeKind> ProcessIncs<VM, KIND> {
    const CAPACITY: usize = crate::args::BUFFER_SIZE;
    const UNLOG_BITS: SideMetadataSpec = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
        .as_spec()
        .extract_side_spec();

    fn worker(&self) -> &'static mut GCWorker<VM> {
        GCWorker::<VM>::current()
    }

    fn copy_context(&self) -> &mut GCWorkerCopyContext<VM> {
        unsafe { &mut *self.copy_context }
    }

    fn __default(lxr: &'static LXR<VM>) -> Self {
        Self {
            incs: vec![],
            inc_slices: vec![],
            new_incs: VectorQueue::default(),
            new_inc_slices: VectorQueue::default(),
            new_incs_count: 0,
            lxr,
            pause: Pause::RefCount,
            in_cm: false,
            no_evac: false,
            depth: 1,
            rc: RefCountHelper::NEW,
            root_kind: None,
            survival_ratio_predictor_local: SurvivalRatioPredictorLocal::default(),
            copy_context: std::ptr::null_mut(),
        }
    }

    fn add_new_slot(&mut self, s: VM::VMSlot) {
        self.new_incs.push(s);
        self.new_incs_count += 1;
        if self.new_incs_count as usize >= Self::CAPACITY {
            self.flush();
        }
    }

    fn add_new_slice(&mut self, s: VM::VMMemorySlice) {
        let len = s.len();
        if self.new_incs_count as usize + len >= Self::CAPACITY {
            self.flush();
        }
        self.new_incs_count += len as u32;
        self.new_inc_slices.push(s);
        if self.new_incs_count as usize >= Self::CAPACITY {
            self.flush();
        }
    }

    pub fn new_objects(_objects: Vec<ObjectReference>) -> Self {
        unreachable!()
    }

    pub fn new(incs: Vec<VM::VMSlot>, lxr: &'static LXR<VM>) -> Self {
        Self {
            incs,
            ..Self::__default(lxr)
        }
    }

    fn promote(&mut self, o: ObjectReference, copied: bool, los: bool, depth: u32) {
        o.verify::<VM>();
        #[cfg(feature = "lxr_rc_trace")]
        if is_rc_traced(o) {
            eprintln!(
                "[rc-trace gc={} promote] {:#x} copied={} los={} depth={}",
                rc_trace_gc(),
                o.to_raw_address(),
                copied,
                los,
                depth
            );
        }
        crate::stat(|s| {
            s.promoted_objects += 1;
            s.promoted_volume += o.get_size::<VM>();
            if self.lxr.los().in_space(o) {
                s.promoted_los_objects += 1;
                s.promoted_los_volume += o.get_size::<VM>();
            }
            if copied {
                s.promoted_copy_objects += 1;
                s.promoted_copy_volume += o.get_size::<VM>();
            }
        });
        let size = o.get_size::<VM>();

        if !los {
            let block = Block::containing(o);
            if !copied && block.is_nursery() {
                block.set_as_in_place_promoted(&self.lxr.immix_space);
            }
            self.rc.promote_with_size(o, size);
            if copied {
                self.survival_ratio_predictor_local
                    .record_copied_promotion(size);
            }
        } else {
            // println!("promote los {:?} {}", o, self.immix().is_marked(o));
        }
        // Don't mark copied objects in initial mark pause. The concurrent marker will do it (and can also resursively mark the old objects).
        if self.in_cm || self.pause == Pause::FinalMark {
            debug_assert!(self.lxr.is_marked(o), "{:?} is not marked", o);
        }
        self.scan_nursery_object(o, los, !copied, depth, size);
    }

    fn record_mature_evac_remset2(
        &mut self,
        slot_in_defrag: bool,
        s: VM::VMSlot,
        o: ObjectReference,
    ) {
        if !(crate::args::RC_MATURE_EVACUATION && (self.in_cm || self.pause == Pause::FinalMark)) {
            return;
        }
        if !slot_in_defrag && self.lxr.in_defrag(o) {
            self.lxr
                .immix_space
                .mature_evac_remset
                .record(s, o, self.lxr);
        }
    }

    fn record_mature_evac_remset(&mut self, s: VM::VMSlot, o: ObjectReference) {
        if !(crate::args::RC_MATURE_EVACUATION && (self.in_cm || self.pause == Pause::FinalMark)) {
            return;
        }
        self.record_mature_evac_remset2(self.lxr.address_in_defrag(s.to_address()), s, o);
    }

    fn scan_nursery_object(
        &mut self,
        o: ObjectReference,
        los: bool,
        in_place_promotion: bool,
        _depth: u32,
        size: usize,
    ) {
        let heap_bytes_per_unlog_byte = if VM::VMObjectModel::COMPRESSED_PTR_ENABLED {
            32usize
        } else {
            64
        };
        let is_val_array = VM::VMScanning::is_val_array(o);
        if los {
            if !is_val_array {
                let start =
                    side_metadata::address_to_meta_address(&Self::UNLOG_BITS, o.to_raw_address())
                        .to_mut_ptr::<u8>();
                let limit = side_metadata::address_to_meta_address(
                    &Self::UNLOG_BITS,
                    (o.to_raw_address() + size).align_up(heap_bytes_per_unlog_byte),
                )
                .to_mut_ptr::<u8>();
                unsafe {
                    let bytes = limit.offset_from(start) as usize;
                    std::ptr::write_bytes(start, 0xffu8, bytes);
                }
            }
            o.to_raw_address().unlog_field_relaxed::<VM>();
        } else if in_place_promotion && !is_val_array {
            let header_size = if VM::VMObjectModel::COMPRESSED_PTR_ENABLED {
                12usize
            } else {
                16
            };
            let step = heap_bytes_per_unlog_byte << 2;
            let end = o.to_raw_address() + size;
            let aligned_end = end.align_up(step);
            let cursor = o.to_raw_address() + header_size;
            let mut cursor = cursor.align_down(step);
            let mut meta = side_metadata::address_to_meta_address(&Self::UNLOG_BITS, cursor);
            while cursor < aligned_end {
                unsafe { meta.store(0xffffffffu32) }
                meta += 4usize;
                cursor += step;
            }
        };
        if VM::VMScanning::is_obj_array(o) && VM::VMScanning::obj_array_data(o).len() > 1024 {
            let data = VM::VMScanning::obj_array_data(o);
            for chunk in data.chunks(Self::CAPACITY) {
                self.add_new_slice(chunk);
            }
        } else if !is_val_array {
            let obj_in_defrag = !los && Block::in_defrag_block::<VM>(o);
            o.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |slot, _| {
                let Some(target) = slot.load() else {
                    return;
                };
                // Misclassification guard (lxr_rc_trace): a value/isbits array
                // scanned as a pointer array will feed data words (Float bits,
                // BitSet masks, poison) here as ObjectReferences.  Detect that
                // at the SOURCE (parent in scope) by validating the loaded
                // child's header word, and log the parent's full array
                // classification so the misclassified array can be identified.
                #[cfg(feature = "lxr_rc_trace")]
                {
                    // Validate the child's Julia TYPE TAG denotes a valid type.
                    // This is the exact check `get_current_size`/
                    // `scan_julia_object` performs (it panics otherwise).  It
                    // reads the tag at `obj - 8` and confirms it resolves to a
                    // real DataType — NOT a check of the first data field at
                    // offset 0 (which legally holds isbits values and would give
                    // false positives).  A `false` result is a genuine
                    // freed/reused object referenced by a live slot = RC
                    // undercount.  Log the parent's full description so the
                    // missing barrier/store can be located.
                    if !VM::VMScanning::debug_object_tag_is_valid(target) {
                        let tag_addr = target.to_raw_address() - 8usize;
                        let tag_word: usize =
                            if tag_addr.is_mapped() { unsafe { tag_addr.load::<usize>() } } else { 0 };
                        let n = CORRUPT_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                        if n < CORRUPT_LOG_LIMIT {
                            eprintln!(
                                "[rc-trace gc={} CORRUPT promo-child-bad-tag #{}] child={:#x} tag={:#x} slot={:?} parent={:#x} parent_desc={}",
                                rc_trace_gc(),
                                n,
                                target.to_raw_address(),
                                tag_word,
                                slot,
                                o.to_raw_address(),
                                VM::VMScanning::debug_describe_object(o),
                            );
                            if n < CORRUPT_SCAN_LIMIT {
                                scan_heap_referrers(target, self.lxr);
                            }
                        }
                    }
                }
                // println!(" -- rec inc opt {:?}.{:?} -> {:?}", o, slot, target);
                debug_assert!(
                    target.to_raw_address().is_mapped(),
                    "Unmapped obj {:?}.{:?} -> {:?}",
                    o,
                    slot,
                    target
                );
                debug_assert!(
                    target.is_in_any_space(),
                    "Unmapped obj {:?}.{:?} -> {:?}",
                    o,
                    slot,
                    target
                );
                // Guard: skip RC ops for objects not in Immix/LOS (no RC_TABLE metadata)
                if !self.object_has_rc_metadata(target) {
                    #[cfg(feature = "lxr_rc_trace")]
                    if is_rc_traced(target) {
                        eprintln!(
                            "[rc-trace gc={} inc-promo-skip] {:#x} parent={:#x} slot={:?} (no RC metadata, e.g. VM/immortal space)",
                            rc_trace_gc(), target.to_raw_address(), o.to_raw_address(), slot
                        );
                    }
                    return;
                }
                let rc = self.rc.count(target);
                if rc == 0 {
                    // println!(" -- rec inc {:?}.{:?} -> {:?}", o, slot, target);
                    #[cfg(feature = "lxr_rc_trace")]
                    if is_rc_traced(target) {
                        eprintln!(
                            "[rc-trace gc={} inc-promo] {:#x} parent={:#x} slot={:?} child_rc=0 (queued for recursive promotion)",
                            rc_trace_gc(), target.to_raw_address(), o.to_raw_address(), slot
                        );
                    }
                    self.add_new_slot(slot);
                } else {
                    if rc != crate::util::rc::MAX_REF_COUNT {
                        #[cfg(feature = "lxr_rc_trace")]
                        if is_rc_traced(target) {
                            eprintln!(
                                "[rc-trace gc={} inc-promo-direct] {:#x} parent={:#x} slot={:?} rc: {} -> {}",
                                rc_trace_gc(), target.to_raw_address(), o.to_raw_address(), slot, rc, rc + 1
                            );
                        }
                        let _ = self.rc.inc(target);
                    }
                    self.record_mature_evac_remset2(obj_in_defrag, slot, target);
                }
            });
        }
    }

    #[cold]
    fn flush(&mut self) {
        if !self.new_incs.is_empty() || !self.new_inc_slices.is_empty() {
            let new_incs = self.new_incs.take();
            let new_inc_slices = self.new_inc_slices.take();
            let mut w = ProcessIncs::<VM, EDGE_KIND_NURSERY>::new(new_incs, self.lxr);
            w.depth += 1;
            w.inc_slices = new_inc_slices;
            self.worker().add_work(WorkBucketStage::Unconstrained, w);
        }
        self.new_incs_count = 0;
    }

    fn inc(&self, o: ObjectReference) -> bool {
        let result = self.rc.inc(o);
        #[cfg(feature = "lxr_rc_trace")]
        if is_rc_traced(o) {
            match result {
                Ok(old_rc) => eprintln!(
                    "[rc-trace gc={} inc] {:#x} rc: {} -> {}",
                    rc_trace_gc(),
                    o.to_raw_address(),
                    old_rc,
                    old_rc + 1
                ),
                Err(old_rc) => eprintln!(
                    "[rc-trace gc={} inc-stuck] {:#x} rc: {} (stuck/max, not incremented)",
                    rc_trace_gc(),
                    o.to_raw_address(),
                    old_rc
                ),
            }
        }
        result == Ok(0)
    }

    fn dont_evacuate(&self, o: ObjectReference, los: bool) -> bool {
        if los {
            return true;
        }
        // Safety check: Never evacuate/forward non-heap/static/system-image objects!
        if !crate::memory_manager::is_in_mmtk_spaces(o) {
            return true;
        }
        // Skip mature object
        if self.rc.count(o) != 0 {
            return true;
        }
        // Skip recycled lines
        if !Block::containing(o).is_nursery() {
            return true;
        }
        if cfg!(debug_assertions) {
            let cls = unsafe { (o.to_raw_address() + 8usize).load::<u32>() };
            assert!(cls != 0, "ERROR {:?} rc={}", o, self.rc.count(o));
        }
        if o.get_size::<VM>() >= crate::args().max_young_evac_size {
            return true;
        }
        false
    }

    /// Check whether an object resides in a space that has RC_TABLE side metadata.
    #[inline]
    fn object_has_rc_metadata(&self, o: ObjectReference) -> bool {
        has_rc_metadata(o, self.lxr)
    }

    fn process_inc_and_evacuate(&mut self, o: ObjectReference, depth: u32) -> ObjectReference {
        o.verify::<VM>();
        // Safe Bounds Guard: If the object is not in any active MMTk space, return immediately.
        // This is 100% bounds-checked and prevents any out-of-bounds SFT dense chunk map segfaults.
        if !crate::memory_manager::is_in_mmtk_spaces(o) {
            return o;
        }
        // Guard: skip RC operations for objects not in Immix or LOS space.
        // RC_TABLE side metadata is only mapped for Immix and LOS.  Objects in
        // VM space (sysimage), immortal space, or non-moving space would SIGSEGV
        // on any RC_TABLE access (inc/dec/count).  These objects have permanent
        // lifetimes and don't need reference counting.
        if !self.object_has_rc_metadata(o) {
            return o;
        }
        let los = self.lxr.los().in_space(o);
        if crate::args::RC_NURSERY_EVACUATION
            && !los
            && object_forwarding::is_forwarded_or_being_forwarded::<VM>(o)
        {
            while object_forwarding::is_being_forwarded::<VM>(o) {
                std::hint::spin_loop();
            }
            let new = if object_forwarding::is_forwarded::<VM>(o) {
                object_forwarding::read_forwarding_pointer::<VM>(o)
            } else {
                o
            };
            crate::stat(|s| {
                s.inc_objects += 1;
                s.inc_volume += new.get_size::<VM>();
            });
            let promoted = self.inc(new);
            if promoted && new == o {
                self.promote(o, false, los, depth);
            }
            return new;
        }
        crate::stat(|s| {
            s.inc_objects += 1;
            s.inc_volume += o.get_size::<VM>();
        });
        if !crate::args::RC_NURSERY_EVACUATION || self.dont_evacuate(o, los) {
            if self.inc(o) {
                self.promote(o, false, los, depth);
            }
            return o;
        }
        let forwarding_status = object_forwarding::attempt_to_forward::<VM>(o);
        if object_forwarding::state_is_forwarded_or_being_forwarded(forwarding_status) {
            // Object is moved to a new location.
            let new = object_forwarding::spin_and_get_forwarded_object::<VM>(o, forwarding_status);
            self.inc(new);
            new
        } else {
            let is_nursery = self.rc.count(o) == 0;
            let copy_depth_reached = crate::args::INC_MAX_COPY_DEPTH && depth > 16;
            if is_nursery && !self.no_evac && !copy_depth_reached {
                // Evacuate the object
                let new = object_forwarding::try_forward_object::<VM>(
                    o,
                    CopySemantics::DefaultCopy,
                    self.copy_context(),
                );
                if let Some(new) = new {
                    self.inc(new);
                    self.promote(new, true, false, depth);
                    new
                } else {
                    gc_log!([1] "to-space overflow");
                    // Object is not moved.
                    let promoted = self.inc(o);
                    object_forwarding::clear_forwarding_bits::<VM>(o);
                    if promoted {
                        self.promote(o, false, los, depth);
                    }
                    crate::NO_EVAC.store(true, Ordering::Relaxed);
                    self.no_evac = true;
                    o
                }
            } else {
                // Object is not moved.
                let promoted = self.inc(o);
                object_forwarding::clear_forwarding_bits::<VM>(o);
                if promoted {
                    self.promote(o, false, los, depth);
                }
                o
            }
        }
    }

    /// Return `None` if the increment of the slot should be delayed
    fn unlog_and_load_rc_object<const K: EdgeKind>(
        &mut self,
        s: VM::VMSlot,
    ) -> Option<ObjectReference> {
        debug_assert!(!crate::args::EAGER_INCREMENTS);
        let o = s.load();
        // unlog slot — but only if the slot address is in the MMTk managed
        // heap.  Non-heap slots (e.g. external GenericMemory data buffers
        // allocated via malloc, with how==1/2) have no side metadata (unlog
        // bits) mapped.  Calling unlog_field_relaxed on such addresses would
        // SIGSEGV.  These slots are enqueued by the non-heap path in
        // enqueue_node (barrier.rs) and are safe to read (the malloc'd
        // buffer remains valid as long as the owning GenericMemory is alive),
        // but must not have their unlog bits touched.
        if K == EDGE_KIND_MATURE {
            use crate::util::heap::layout::vm_layout::vm_layout;
            let slot_addr = s.to_address();
            let layout = vm_layout();
            if slot_addr >= layout.heap_start && slot_addr < layout.heap_end {
                slot_addr.unlog_field_relaxed::<VM>();
            }
        }
        o
    }

    fn process_slot<const K: EdgeKind>(
        &mut self,
        s: VM::VMSlot,
        depth: u32,
        add_root_to_remset: bool,
    ) -> Option<ObjectReference> {
        #[cfg(feature = "lxr_rc_trace")]
        if is_rc_traced_addr(s.to_address().as_usize()) {
            let kind_str = match K {
                EDGE_KIND_ROOT => "ROOT",
                EDGE_KIND_NURSERY => "NURSERY",
                EDGE_KIND_MATURE => "MATURE",
                _ => "UNKNOWN",
            };
            let loaded_o = s.load();
            let loaded = loaded_o.map(|o| o.to_raw_address().as_usize()).unwrap_or(0);
            let lrc = if let Some(o) = loaded_o {
                if self.object_has_rc_metadata(o) {
                    self.rc.count(o) as isize
                } else {
                    -1
                }
            } else {
                -2
            };
            eprintln!(
                "[rc-trace gc={} process-slot] slot={:#x} kind={} loaded={:#x} rc={} depth={}",
                rc_trace_gc(),
                s.to_address().as_usize(),
                kind_str,
                loaded,
                lrc,
                depth
            );
        }
        let o = match self.unlog_and_load_rc_object::<K>(s) {
            Some(o) => o,
            _ => {
                return None;
            }
        };
        // println!(" - inc {:?}: {:?} rc={}", s, o, self.rc.count(o));
        o.verify::<VM>();
        #[cfg(feature = "lxr_rc_trace")]
        if is_rc_traced(o) {
            let kind_str = match K {
                EDGE_KIND_ROOT => "ROOT",
                EDGE_KIND_NURSERY => "NURSERY",
                EDGE_KIND_MATURE => "MATURE",
                _ => "UNKNOWN",
            };
            let rc_before = if self.object_has_rc_metadata(o) {
                self.rc.count(o) as isize
            } else {
                -1 // not in RC space
            };
            eprintln!(
                "[rc-trace gc={} inc-slot] {:#x} slot={:?} kind={} rc_before={} depth={}",
                rc_trace_gc(),
                o.to_raw_address(),
                s,
                kind_str,
                rc_before,
                depth
            );
        }
        // Corruption guard (lxr_rc_trace): before processing an inc on `o`,
        // validate that `o`'s Julia TYPE TAG (vtag) denotes a real DataType.
        //
        // The vtag lives at `o - sizeof(jl_taggedvalue_t)` (= o - 8), NOT at
        // offset 0 (which is the object's first DATA field and may legally hold
        // any isbits value such as 0xffff... for a BitSet chunk or a Float bit
        // pattern — reading offset 0 produces false positives; §2.16's
        // "value-array scanned as pointers" claim was exactly that artifact).
        //
        // §2.17: the previous guard here used a LAX plausibility check ("vt is
        // small OR a mapped pointer") which passes tags that are mapped but do
        // NOT denote a DataType — exactly the case that later crashes
        // `get_current_size` with `!jl_is_datatype(vt)`.  We now use the EXACT
        // validity check `get_current_size`/`scan_julia_object` use, exposed via
        // `VMScanning::debug_object_tag_is_valid` (resolve forwarding → confirm
        // the tag points to a `jl_datatype_t` whose own tag is the DataType
        // small-tag and `smalltag()==0`).  This catches the genuine corruption
        // AT the inc-slot — before the panic — for ALL edge kinds, INCLUDING
        // `EDGE_KIND_ROOT` (the dangling object is reached as a deep queued
        // NURSERY/ROOT edge whose storing parent is not yet identified).
        //
        // It logs the offending object's tag-at-−8, the slot address, and the
        // edge kind so the slot can be fed to MMTK_RC_TRACE_ADDRS to find the
        // missing inc/barrier on the dangling object.  Zero cost when feature
        // off.
        #[cfg(feature = "lxr_rc_trace")]
        {
            if !VM::VMScanning::debug_object_tag_is_valid(o) {
                // Rate-limit: the per-event heap-wide referrer scan is O(heap)
                // and there can be many events, so only do the full scan for
                // the first CORRUPT_SCAN_LIMIT events; afterwards just log.
                let n = CORRUPT_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                if n < CORRUPT_LOG_LIMIT {
                    let tag_addr = o.to_raw_address() - 8usize;
                    let tag: usize = if tag_addr.is_mapped() {
                        unsafe { tag_addr.load::<usize>() }
                    } else {
                        0
                    };
                    let kind_str = match K {
                        EDGE_KIND_ROOT => "ROOT",
                        EDGE_KIND_NURSERY => "NURSERY",
                        EDGE_KIND_MATURE => "MATURE",
                        _ => "UNKNOWN",
                    };
                    eprintln!(
                        "[rc-trace gc={} CORRUPT inc-on-bad-tag #{}] obj={:#x} tag@-8={:#x} rc={} slot={:#x} slot_dbg={:?} kind={} depth={} desc={} — live slot references a FREED/corrupt object",
                        rc_trace_gc(),
                        n,
                        o.to_raw_address(),
                        tag,
                        if self.object_has_rc_metadata(o) { self.rc.count(o) as isize } else { -1 },
                        s.to_address().as_usize(),
                        s,
                        kind_str,
                        depth,
                        VM::VMScanning::debug_describe_object(o),
                    );
                    if n < CORRUPT_SCAN_LIMIT {
                        // Find who else points at this corrupt object.
                        scan_heap_referrers(o, self.lxr);
                    }
                }
            }
        }
        let new = self.process_inc_and_evacuate(o, depth);
        if s.is_type_tag() {
            let _ = self.rc.stick(new);
        }
        // Put this into remset if this is a mature slot, or a weak root
        if K != EDGE_KIND_ROOT || add_root_to_remset {
            self.record_mature_evac_remset(s, new);
        }
        if new != o {
            // gc_log!(
            //     " -- inc {:?}: {:?} => {:?} rc={} {:?}",
            //     s,
            //     o,
            //     new.range::<VM>(),
            //     self.rc.count(new),
            //     K
            // );
            s.store(new)
        } else {
            // gc_log!(
            //     " -- inc {:?}: {:?} rc={} {:?}",
            //     s,
            //     o.range::<VM>(),
            //     self.rc.count(o),
            //     K
            // );
        }
        Some(new)
    }

    #[inline]
    fn prefetch_object(&self, o: ObjectReference) {
        prefetch_object(o, &self.rc);
    }

    fn process_incs<const K: EdgeKind>(
        &mut self,
        mut incs: AddressBuffer<'_, VM::VMSlot>,
        depth: u32,
        add_root_to_remset: bool,
    ) -> Option<Vec<ObjectReference>> {
        if K == EDGE_KIND_ROOT {
            let roots = incs.as_mut_ptr() as *mut ObjectReference;
            let mut num_roots = 0usize;
            for (i, s) in incs.iter().enumerate() {
                if let Some(new) = self.process_slot::<K>(*s, depth, add_root_to_remset) {
                    unsafe {
                        roots.add(num_roots).write(new);
                    }
                    num_roots += 1;
                }
                if crate::args::PREFETCH {
                    if let Some(s) = incs.get(i + crate::args::PREFETCH_STEP) {
                        if let Some(o) = s.load() {
                            self.prefetch_object(o);
                        }
                    }
                }
            }
            if num_roots != 0 {
                let cap = incs.capacity();
                std::mem::forget(incs);
                let roots =
                    unsafe { Vec::<ObjectReference>::from_raw_parts(roots, num_roots, cap) };
                Some(roots)
            } else {
                None
            }
        } else {
            for (i, s) in incs.iter().enumerate() {
                self.process_slot::<K>(*s, depth, false);
                if crate::args::PREFETCH {
                    if let Some(s) = incs.get(i + crate::args::PREFETCH_STEP) {
                        if let Some(o) = s.load() {
                            self.prefetch_object(o);
                        }
                    }
                }
            }
            None
        }
    }

    fn process_incs_for_obj_array<const K: EdgeKind>(
        &mut self,
        slice: VM::VMMemorySlice,
        depth: u32,
    ) -> Option<Vec<ObjectReference>> {
        let n = slice.len();
        for (i, s) in slice.iter_slots().enumerate() {
            self.process_slot::<K>(s, depth, false);
            if crate::args::PREFETCH {
                if i + crate::args::PREFETCH_STEP < n {
                    let s = slice.get(i + crate::args::PREFETCH_STEP);
                    if let Some(o) = s.load() {
                        self.prefetch_object(o);
                    }
                }
            }
        }
        None
    }
}

pub type EdgeKind = u8;
pub const EDGE_KIND_ROOT: u8 = 0;
pub const EDGE_KIND_NURSERY: u8 = 1;
pub const EDGE_KIND_MATURE: u8 = 2;

enum AddressBuffer<'a, S: Slot> {
    Owned(Vec<S>),
    Ref(&'a mut Vec<S>),
}

impl<S: Slot> Deref for AddressBuffer<'_, S> {
    type Target = Vec<S>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(x) => x,
            Self::Ref(x) => x,
        }
    }
}

impl<S: Slot> DerefMut for AddressBuffer<'_, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(x) => x,
            Self::Ref(x) => x,
        }
    }
}

impl<VM: VMBinding, const KIND: EdgeKind> GCWork<VM> for ProcessIncs<VM, KIND> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        self.lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        self.pause = self.lxr.current_pause().unwrap();
        self.in_cm = self.lxr.cm_in_progress();
        self.copy_context = self.worker().get_copy_context_mut() as *mut GCWorkerCopyContext<VM>;
        if crate::NO_EVAC.load(Ordering::Relaxed) {
            self.no_evac = true;
        } else {
            let over_time = crate::args()
                .max_pause_millis
                .map(|threshold| crate::GC_START_TIME.elapsed().as_millis() >= threshold as u128)
                .unwrap_or(false);
            let over_space = mmtk.get_plan().get_used_pages()
                - mmtk.get_plan().get_collection_reserved_pages()
                > mmtk.get_plan().get_total_pages();
            if over_space || over_time {
                self.no_evac = true;
                crate::NO_EVAC.store(true, Ordering::Relaxed);
                gc_log!([2]
                    " - Stop evacuation. over_space={} over_time={}",
                    over_space,
                    over_time
                );
            }
        }
        // Process main buffer
        let root_slots = if KIND == EDGE_KIND_ROOT
            && (self.pause == Pause::FinalMark || self.pause == Pause::Full)
        {
            self.incs.clone()
        } else {
            vec![]
        };
        let add_root_to_remset = self
            .root_kind
            .map(|r| r.should_record_remset())
            .unwrap_or_default();
        let roots = {
            let incs = std::mem::take(&mut self.incs);
            self.process_incs::<KIND>(AddressBuffer::Owned(incs), self.depth, false)
        };
        if cfg!(debug_assertions) && !self.inc_slices.is_empty() {
            assert!(!add_root_to_remset);
        }
        for s in std::mem::take(&mut self.inc_slices) {
            self.process_incs_for_obj_array::<KIND>(s, self.depth);
        }
        if let Some(roots) = roots {
            if self.lxr.cm_enabled()
                && self.pause == Pause::InitialMark
                && !self.root_kind.unwrap().should_skip_mark_and_decs()
            {
                if cfg!(any(feature = "sanity", debug_assertions)) {
                    for r in &roots {
                        assert!(
                            r.to_raw_address().is_mapped(),
                            "Invalid object {:?}: address is not mapped",
                            r
                        );
                    }
                }
                worker
                    .scheduler()
                    .postpone(LXRConcurrentTraceObjects::new(roots.clone(), mmtk));
            }
            if self.pause == Pause::FinalMark || self.pause == Pause::Full {
                if !root_slots.is_empty() && self.root_kind != Some(RootKind::Weak) {
                    if self.pause == Pause::FinalMark {
                        let mut w = LXRStopTheWorldProcessEdges::<_, false>::new(
                            root_slots,
                            true,
                            mmtk,
                            WorkBucketStage::Closure,
                        );
                        w.root_kind = self.root_kind;
                        worker.add_work(WorkBucketStage::Closure, w)
                    } else {
                        let mut w = LXRStopTheWorldProcessEdges::<_, true>::new(
                            root_slots,
                            true,
                            mmtk,
                            WorkBucketStage::Closure,
                        );
                        w.root_kind = self.root_kind;
                        worker.add_work(WorkBucketStage::Closure, w)
                    };
                }
            } else if !self.root_kind.unwrap().should_skip_decs() {
                self.lxr.curr_roots.read().unwrap().push(roots);
            }
        }
        // Process recursively generated buffer
        let mut depth = self.depth;
        let mut incs = vec![];
        let mut inc_slices = vec![];
        const ACTIVE_PACKET_SPLIT: bool = false;
        while !self.new_incs.is_empty() || !self.new_inc_slices.is_empty() {
            self.new_incs_count = 0;
            depth += 1;
            incs.clear();
            inc_slices.clear();
            self.new_incs.swap(&mut incs);
            self.new_inc_slices.swap(&mut inc_slices);
            if ACTIVE_PACKET_SPLIT && depth >= 16 && incs.len() > 1 {
                let (a, b) = incs.split_at(incs.len() / 2);
                let mut w = ProcessIncs::<VM, EDGE_KIND_NURSERY>::new(b.to_vec(), self.lxr);
                w.depth = depth;
                self.worker().add_work(WorkBucketStage::Unconstrained, w);
                incs = a.to_vec();
            }
            if !incs.is_empty() {
                self.process_incs::<EDGE_KIND_NURSERY>(AddressBuffer::Ref(&mut incs), depth, false);
            }
            if !inc_slices.is_empty() {
                for s in &inc_slices {
                    self.process_incs_for_obj_array::<EDGE_KIND_NURSERY>(s.clone(), self.depth);
                }
            }
        }
        self.survival_ratio_predictor_local.sync();
    }
}

pub struct ProcessDecs<VM: VMBinding> {
    /// Decrements to process
    decs: Option<Vec<ObjectReference>>,
    decs_arc: Option<Arc<Vec<ObjectReference>>>,
    /// Recursively generated new decrements
    new_decs: VectorQueue<ObjectReference>,
    counter: LazySweepingJobsCounter,
    mark_objects: VectorQueue<ObjectReference>,
    mark_dead_objects: bool,
    cld_policy: CLDScanPolicy,
    mature_sweeping_in_progress: bool,
    rc: RefCountHelper<VM>,
}

impl<VM: VMBinding> ProcessDecs<VM> {
    pub const CAPACITY: usize = crate::args::BUFFER_SIZE;

    fn worker(&self) -> &mut GCWorker<VM> {
        GCWorker::<VM>::current()
    }

    pub fn new(decs: Vec<ObjectReference>, counter: LazySweepingJobsCounter) -> Self {
        Self {
            decs: Some(decs),
            decs_arc: None,
            new_decs: VectorQueue::default(),
            counter,
            mark_objects: VectorQueue::default(),
            mark_dead_objects: false,
            cld_policy: CLDScanPolicy::Ignore,
            mature_sweeping_in_progress: false,
            rc: RefCountHelper::NEW,
        }
    }

    pub fn new_arc(decs: Arc<Vec<ObjectReference>>, counter: LazySweepingJobsCounter) -> Self {
        Self {
            decs: None,
            decs_arc: Some(decs),
            new_decs: VectorQueue::default(),
            counter,
            mark_objects: VectorQueue::default(),
            mark_dead_objects: false,
            cld_policy: CLDScanPolicy::Ignore,
            mature_sweeping_in_progress: false,
            rc: RefCountHelper::NEW,
        }
    }

    fn recursive_dec(&mut self, o: ObjectReference) {
        self.new_decs.push(o);
        if self.new_decs.is_full() {
            self.flush()
        }
    }

    fn new_work(&self, lxr: &LXR<VM>, w: ProcessDecs<VM>) {
        if lxr.current_pause().is_none() {
            self.worker()
                .add_work_prioritized(WorkBucketStage::Unconstrained, w);
        } else {
            self.worker().add_work(WorkBucketStage::Unconstrained, w);
        }
    }

    fn flush(&mut self) {
        let mmtk = GCWorker::<VM>::current().mmtk;
        if !self.new_decs.is_empty() {
            let new_decs = self.new_decs.take();
            let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
            self.new_work(
                lxr,
                ProcessDecs::new(new_decs, self.counter.clone_with_decs()),
            );
        }
        if !self.mark_objects.is_empty() {
            let objects = self.mark_objects.take();
            let w = LXRConcurrentTraceObjects::new(objects, mmtk);
            if crate::args::LAZY_DECREMENTS {
                self.worker().add_work(WorkBucketStage::Unconstrained, w);
            } else {
                self.worker().scheduler().postpone(w);
            }
        }
    }

    fn record_mature_evac_remset(&mut self, lxr: &LXR<VM>, s: VM::VMSlot, o: ObjectReference) {
        if !(crate::args::RC_MATURE_EVACUATION && self.mark_dead_objects) {
            return;
        }
        if !lxr.address_in_defrag(s.to_address()) && lxr.in_defrag(o) {
            lxr.immix_space.mature_evac_remset.record(s, o, lxr);
        }
    }

    #[cold]
    fn process_dead_object(&mut self, o: ObjectReference, lxr: &LXR<VM>) -> bool {
        // === Per-object RC trace: death event ===
        #[cfg(feature = "lxr_rc_trace")]
        if is_rc_traced(o) {
            let obj_addr = o.to_raw_address();
            let vtag: usize = if (obj_addr - 8usize).is_mapped() {
                unsafe { (obj_addr - 8usize).load::<usize>() }
            } else {
                0xDEAD
            };
            let size = if obj_addr.is_mapped() {
                o.get_size::<VM>()
            } else {
                0
            };
            eprintln!(
                "[rc-trace gc={} death] {:#x} vtag={:#x} size={} — THIS OBJECT IS BEING FREED",
                rc_trace_gc(),
                obj_addr,
                vtag,
                size
            );
            // Reverse-reference scan: walk the entire live heap (Immix + LOS)
            // and report every live object that still holds a pointer to the
            // dying object.  This deterministically enumerates the heap
            // referrers of the dying object — i.e. the "untracked referrer"
            // the §2.15 handoff asks us to find.  If this reports ZERO heap
            // referrers, the dangling pointer is a C stack local (a missing
            // GC root), not a missing write barrier.
            scan_heap_referrers(o, lxr);
        }
        // === Death-time genuine-undercount detector (§2.17) ===
        // For EVERY death (not just MMTK_RC_TRACE_ADDRS ones), if enabled via
        // MMTK_RC_UNDERCOUNT_BUDGET/_FROM_GC, scan the live heap for a
        // referrer; report only deaths that still have a LIVE referrer (the
        // genuine RC undercount).  Near-zero cost when disabled.
        #[cfg(feature = "lxr_rc_trace")]
        detect_death_undercount(o, lxr);
        // === RC death diagnostic instrumentation ===
        // Gated behind the `lxr_rc_death_diag` feature flag to avoid
        // eprintln! calls on the hot path in production builds.
        // Enable with: --features lxr_rc_death_diag
        #[cfg(feature = "lxr_rc_death_diag")]
        {
            let n = DEATH_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let gc_cycle = GC_CYCLE_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            if n < DEATH_LOG_LIMIT {
                let obj_addr = o.to_raw_address();
                let in_immix = lxr.immix_space.in_space(o);
                let in_los = lxr.los().in_space(o);
                let space = if in_immix {
                    "immix"
                } else if in_los {
                    "LOS"
                } else {
                    "???"
                };
                // Read vtag at obj - 8 (Julia header word)
                let vtag_addr = obj_addr - 8usize;
                let vtag: usize = if vtag_addr.is_mapped() {
                    unsafe { vtag_addr.load::<usize>() }
                } else {
                    0xDEAD
                };
                let size = if obj_addr.is_mapped() {
                    o.get_size::<VM>()
                } else {
                    0
                };
                eprintln!(
                    "[rc-death #{:>4} gc={}] addr={:#x} vtag={:#x} size={} space={}",
                    n, gc_cycle, obj_addr, vtag, size, space
                );
                // For 48-byte objects (likely BindingPartition), dump pointer fields
                // to help trace the ownership chain.
                // BindingPartition layout (Julia 1.12, from julia.h):
                //   +0:  restriction (jl_value_t*)
                //   +8:  min_world (_Atomic size_t)
                //   +16: max_world (_Atomic size_t)
                //   +24: next (_Atomic jl_binding_partition_t*)
                //   +32: kind (size_t)
                if size == 48 && obj_addr.is_mapped() {
                    let restriction: usize = unsafe { obj_addr.load::<usize>() };
                    let min_world: usize = unsafe { (obj_addr + 8usize).load::<usize>() };
                    let max_world: usize = unsafe { (obj_addr + 16usize).load::<usize>() };
                    let next: usize = unsafe { (obj_addr + 24usize).load::<usize>() };
                    let kind: usize = unsafe { (obj_addr + 32usize).load::<usize>() };
                    // Check if restriction or next point to objects with valid RC
                    let restriction_rc = if restriction != 0 {
                        let robj = unsafe {
                            ObjectReference::from_raw_address_unchecked(
                                crate::util::Address::from_usize(restriction),
                            )
                        };
                        if has_rc_metadata(robj, lxr) {
                            self.rc.count(robj) as isize
                        } else {
                            -1 // not in RC space
                        }
                    } else {
                        -2 // NULL
                    };
                    let next_rc = if next != 0 {
                        let nobj = unsafe {
                            ObjectReference::from_raw_address_unchecked(
                                crate::util::Address::from_usize(next),
                            )
                        };
                        if has_rc_metadata(nobj, lxr) {
                            self.rc.count(nobj) as isize
                        } else {
                            -1 // not in RC space
                        }
                    } else {
                        -2 // NULL
                    };
                    eprintln!(
                        "         restriction={:#x}(rc={}) kind={:#x} worlds=[{},{}] next={:#x}(rc={})",
                        restriction, restriction_rc, kind, min_world, max_world, next, next_rc
                    );
                }
            } else if n == DEATH_LOG_LIMIT {
                eprintln!(
                    "[rc-death] ... suppressing further death logs (limit={}) at gc={}",
                    DEATH_LOG_LIMIT, gc_cycle
                );
            }
        }
        // === end diagnostic ===

        crate::stat(|s| {
            s.dead_mature_objects += 1;
            s.dead_mature_volume += o.get_size::<VM>();

            s.dead_mature_rc_objects += 1;
            s.dead_mature_rc_volume += o.get_size::<VM>();

            if !lxr.immix_space.in_space(o) {
                s.dead_mature_los_objects += 1;
                s.dead_mature_los_volume += o.get_size::<VM>();

                s.dead_mature_rc_los_objects += 1;
                s.dead_mature_rc_los_volume += o.get_size::<VM>();
            }
        });
        if self.mark_dead_objects {
            lxr.mark(o);
        }
        // Recursively decrease field ref counts
        o.iterate_fields::<VM, _>(
            self.cld_policy,
            RefScanPolicy::Follow,
            |slot, out_of_heap| {
                if let Some(x) = slot.load() {
                    // println!(" -- rec dec {:?}.{:?} -> {:?}", o, slot, x);
                    if !out_of_heap {
                        // Guard: skip RC ops for objects not in Immix/LOS (no RC_TABLE metadata)
                        if has_rc_metadata(x, lxr) {
                            let rc = self.rc.count(x);
                            if rc != MAX_REF_COUNT && rc != 0 {
                                #[cfg(feature = "lxr_rc_trace")]
                                if is_rc_traced(x) {
                                    eprintln!(
                                        "[rc-trace gc={} rec-dec] {:#x} from dying parent={:#x} slot={:?} child_rc={}",
                                        rc_trace_gc(), x.to_raw_address(), o.to_raw_address(), slot, rc
                                    );
                                }
                                self.recursive_dec(x);
                            }
                        }
                    } else {
                        self.record_mature_evac_remset(lxr, slot, x);
                    }
                    // Guard: only check/set mark bits for objects in Immix/LOS.
                    // VM-space and immortal-space objects have no side metadata
                    // mapped — calling is_marked() or attempt_mark() on them
                    // accesses unmapped pages → SIGSEGV.
                    if self.mark_dead_objects && has_rc_metadata(x, lxr) && !lxr.is_marked(x) {
                        if cfg!(any(feature = "sanity", debug_assertions)) {
                            assert!(
                                x.to_raw_address().is_mapped(),
                                "Invalid object {:?}.{:?} -> {:?}: address is not mapped",
                                o,
                                slot,
                                x
                            );
                        }
                        self.mark_objects.push(x);
                        if self.mark_objects.is_full() {
                            self.flush();
                        }
                    }
                }
            },
        );
        let in_ix_space = lxr.immix_space.in_space(o);
        if in_ix_space {
            self.rc.unmark_straddle_object(o);
        }
        if cfg!(feature = "sanity") || ObjectReference::STRICT_VERIFICATION {
            unsafe { o.to_raw_address().store(0xdeadusize) };
        }
        if in_ix_space {
            let block = Block::containing(o);
            lxr.immix_space
                .add_to_possibly_dead_mature_blocks(block, false);
            false
        } else {
            true
        }
    }

    #[inline]
    fn prefetch_object(&self, o: ObjectReference) {
        prefetch_object(o, &self.rc);
    }

    fn process_decs(&mut self, decs: &[ObjectReference], lxr: &LXR<VM>) {
        for (i, o) in decs.iter().enumerate() {
            // Inline space membership once — avoids redundant chunk descriptor
            // lookups that would occur if we called has_rc_metadata() and then
            // los.in_space() separately.  Short-circuits: if in_immix (the
            // common case), los.in_space is never evaluated.
            let in_immix = lxr.immix_space.in_space(*o);
            let in_los = !in_immix && lxr.los().in_space(*o);
            if !in_immix && !in_los {
                // Not in any RC-tracked space (VM space, immortal, etc.)
                #[cfg(feature = "lxr_rc_trace")]
                if is_rc_traced(*o) {
                    eprintln!(
                        "[rc-trace gc={} dec-skip] {:#x} (not in Immix/LOS — no RC metadata)",
                        rc_trace_gc(),
                        o.to_raw_address()
                    );
                }
                continue;
            }
            if self.rc.is_dead_or_stuck(*o)
                || (self.mature_sweeping_in_progress && !lxr.is_marked(*o))
            {
                #[cfg(feature = "lxr_rc_trace")]
                if is_rc_traced(*o) {
                    let rc_val = self.rc.count(*o);
                    let is_dead = self.rc.is_dead_or_stuck(*o);
                    let not_marked = self.mature_sweeping_in_progress && !lxr.is_marked(*o);
                    eprintln!(
                        "[rc-trace gc={} dec-skip] {:#x} rc={} dead_or_stuck={} mature_sweep_unmarked={}",
                        rc_trace_gc(), o.to_raw_address(), rc_val, is_dead, not_marked
                    );
                }
                continue;
            }
            // Guard: only Immix objects have LOCAL_FORWARDING_BITS_SPEC metadata.
            // LOS objects pass the RC metadata check (they have RC_TABLE) but do
            // NOT have forwarding-bits metadata mapped.  Calling is_forwarded()
            // on a LOS object reads unmapped side metadata → SIGSEGV.
            let o = if crate::args::RC_MATURE_EVACUATION
                && in_immix
                && object_forwarding::is_forwarded::<VM>(*o)
            {
                object_forwarding::read_forwarding_pointer::<VM>(*o)
            } else {
                *o
            };
            #[cfg(feature = "lxr_rc_trace")]
            let rc_before_dec = if is_rc_traced(o) { self.rc.count(o) } else { 0 };
            let mut dead = false;
            let mut is_los = false;
            let result = self.rc.clone().fetch_update(o, |c| {
                if c == 1 && !dead {
                    dead = true;
                    is_los = self.process_dead_object(o, lxr);
                }
                debug_assert!(c <= MAX_REF_COUNT);
                if c == 0 || c == MAX_REF_COUNT {
                    None /* sticky */
                } else {
                    Some(c - 1)
                }
            });
            #[cfg(feature = "lxr_rc_trace")]
            if is_rc_traced(o) {
                match result {
                    Ok(old_rc) => eprintln!(
                        "[rc-trace gc={} dec] {:#x} rc: {} -> {} dead={}",
                        rc_trace_gc(), o.to_raw_address(), old_rc, old_rc - 1, dead
                    ),
                    Err(old_rc) => eprintln!(
                        "[rc-trace gc={} dec-stuck] {:#x} rc: {} (stuck/zero, not decremented) dead={}",
                        rc_trace_gc(), o.to_raw_address(), old_rc, dead
                    ),
                }
            }
            if result == Ok(1) && is_los {
                lxr.los().rc_free(o);
            }
            if crate::args::PREFETCH {
                if let Some(o) = decs.get(i + crate::args::PREFETCH_STEP) {
                    self.prefetch_object(*o);
                }
            }
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for ProcessDecs<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        // Increment GC cycle counter on the first ProcessDecs packet of each cycle.
        // This is approximate (multiple packets per cycle) but sufficient for diagnostics.
        #[cfg(feature = "lxr_rc_death_diag")]
        GC_CYCLE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "lxr_rc_trace")]
        rc_trace_inc_gc_count();
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        self.mark_dead_objects = if crate::args::LAZY_DECREMENTS {
            lxr.cm_in_progress() && lxr.previous_pause() != Some(Pause::InitialMark)
        } else {
            lxr.cm_in_progress() && lxr.current_pause() != Some(Pause::InitialMark)
        };
        self.mature_sweeping_in_progress = if crate::args::LAZY_DECREMENTS {
            lxr.previous_pause() == Some(Pause::FinalMark)
                || lxr.current_pause() == Some(Pause::Full)
        } else {
            lxr.current_pause() == Some(Pause::FinalMark)
                || lxr.current_pause() == Some(Pause::Full)
        };
        if let Some(decs) = std::mem::take(&mut self.decs) {
            self.process_decs(&decs, lxr);
        } else if let Some(decs) = std::mem::take(&mut self.decs_arc) {
            self.process_decs(&decs, lxr);
        }
        let mut decs = vec![];
        while !self.new_decs.is_empty() {
            decs.clear();
            self.new_decs.swap(&mut decs);
            self.process_decs(&decs, lxr);
        }
        self.flush();
    }
}

pub struct RCImmixCollectRootEdges<VM: VMBinding> {
    base: ProcessEdgesBase<VM>,
}

impl<VM: VMBinding> ProcessEdgesWork for RCImmixCollectRootEdges<VM> {
    type VM = VM;
    type ScanObjectsWorkType = ScanObjects<Self>;
    const OVERWRITE_REFERENCE: bool = false;
    const RC_ROOTS: bool = true;
    const SCAN_OBJECTS_IMMEDIATELY: bool = true;

    fn new(
        slots: Vec<SlotOf<Self>>,
        roots: bool,
        mmtk: &'static MMTK<VM>,
        bucket: WorkBucketStage,
    ) -> Self {
        debug_assert!(roots);
        let base = ProcessEdgesBase::new(slots, roots, mmtk, bucket);
        Self { base }
    }

    fn trace_object(&mut self, _object: ObjectReference) -> ObjectReference {
        unreachable!()
    }

    fn process_slots(&mut self) {
        if !self.slots.is_empty() {
            #[cfg(feature = "sanity")]
            if self.roots
                && !self.mmtk().get_plan().is_in_sanity()
                && self.root_kind != Some(RootKind::Weak)
            {
                self.cache_roots_for_sanity_gc(self.slots.clone());
            }
            let lxr = self.mmtk().get_plan().downcast_ref::<LXR<VM>>().unwrap();
            let roots = std::mem::take(&mut self.slots);
            let mut w = ProcessIncs::<_, EDGE_KIND_ROOT>::new(roots, lxr);
            w.root_kind = self.root_kind;
            GCWork::do_work(&mut w, self.worker(), self.mmtk());
        }
    }

    fn create_scan_work(&self, _nodes: Vec<ObjectReference>) -> Self::ScanObjectsWorkType {
        unimplemented!()
    }
}

impl<VM: VMBinding> Deref for RCImmixCollectRootEdges<VM> {
    type Target = ProcessEdgesBase<VM>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<VM: VMBinding> DerefMut for RCImmixCollectRootEdges<VM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

// NOTE: A remembered non-heap-buffer owner re-scan (HANDOFF §2.13 option (b))
// was prototyped here and removed in favour of the LOS-routing fix: pointer-
// bearing GenericMemory buffers are now allocated inline in the MMTk heap (LOS),
// so every reference slot is heap-resident with side metadata and the field-
// logging barrier works unmodified.  See plan-alloc.md (Phase 1).  Do not
// re-introduce an owner-rescan / `nonheap_owners` mechanism.
