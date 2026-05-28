//! Read/Write barrier implementations.

use std::sync::Arc;

use atomic::Ordering;

use super::LXR;
use crate::plan::barriers::BarrierSemantics;
use crate::plan::barriers::LOGGED_VALUE;
use crate::plan::barriers::UNLOGGED_VALUE;
use crate::plan::barriers::{FAST_COUNT, SLOW_COUNT};
use crate::plan::immix::Pause;
use crate::plan::lxr::cm::ProcessModBufSATB;
use crate::plan::lxr::rc::ProcessDecs;
use crate::plan::lxr::rc::ProcessIncs;
use crate::plan::lxr::rc::EDGE_KIND_MATURE;
use crate::plan::VectorQueue;
use crate::scheduler::WorkBucketStage;
use crate::util::address::CLDScanPolicy;
use crate::util::address::RefScanPolicy;
use crate::util::metadata::side_metadata::address_to_meta_address;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::*;
use crate::vm::slot::MemorySlice;
use crate::vm::slot::Slot;
use crate::vm::*;
use crate::LazySweepingJobsCounter;
use crate::MMTK;

pub const TAKERATE_MEASUREMENT: bool = false;

pub struct LXRFieldBarrierSemantics<VM: VMBinding> {
    mmtk: &'static MMTK<VM>,
    incs: VectorQueue<VM::VMSlot>,
    decs: VectorQueue<ObjectReference>,
    refs: VectorQueue<ObjectReference>,
    lxr: &'static LXR<VM>,
}

impl<VM: VMBinding> LXRFieldBarrierSemantics<VM> {
    const UNLOG_BITS: SideMetadataSpec = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
        .as_spec()
        .extract_side_spec();

    #[allow(unused)]
    pub fn new(mmtk: &'static MMTK<VM>) -> Self {
        Self {
            mmtk,
            incs: VectorQueue::default(),
            decs: VectorQueue::default(),
            refs: VectorQueue::default(),
            lxr: mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap(),
        }
    }

    fn get_slot_logging_state(&self, slot: VM::VMSlot) -> u8 {
        unsafe { Self::UNLOG_BITS.load(slot.to_address()) }
    }

    fn attempt_to_log_field(&self, slot: VM::VMSlot) -> bool {
        loop {
            // Bailout if logged
            if self.get_slot_logging_state(slot) == LOGGED_VALUE {
                return false;
            }
            // Attempt to log the slots
            match Self::UNLOG_BITS.compare_exchange_atomic(
                slot.to_address(),
                UNLOGGED_VALUE,
                LOGGED_VALUE,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(current) => {
                    if current == LOGGED_VALUE {
                        return false;
                    }
                }
            }
            // Failed to log the slot. Spin.
            std::hint::spin_loop();
        }
    }

    fn log_slot_and_get_old_target(&self, slot: VM::VMSlot) -> Result<Option<ObjectReference>, ()> {
        if self.get_slot_logging_state(slot) == LOGGED_VALUE {
            return Err(());
        }
        let old = slot.load();
        if self.attempt_to_log_field(slot) {
            Ok(old)
        } else {
            Err(())
        }
    }

    fn slow(
        &mut self,
        _src: Option<ObjectReference>,
        slot: VM::VMSlot,
        old: Option<ObjectReference>,
    ) {
        // Reference counting
        if let Some(old) = old {
            self.decs.push(old);
            if self.decs.is_full() {
                self.flush_decs_and_satb();
            }
        }
        self.incs.push(slot);
        if self.incs.is_full() {
            self.flush_incs();
        }
    }

    fn enqueue_node(
        &mut self,
        src: Option<ObjectReference>,
        slot: VM::VMSlot,
        _new: Option<ObjectReference>,
    ) -> bool {
        // Check whether the slot address is in the MMTk managed heap.
        // Slots in malloc'd memory (e.g. external GenericMemory data buffers
        // with how==1/2) have no side metadata (unlog bits) mapped.
        let in_heap = {
            use crate::util::heap::layout::vm_layout::vm_layout;
            let slot_addr = slot.to_address();
            let layout = vm_layout();
            slot_addr >= layout.heap_start && slot_addr < layout.heap_end
        };
        if in_heap {
            // Standard path for in-heap slots: use the unlog bit to
            // deduplicate barrier fires within a GC cycle.  Only the first
            // write to each slot fires the slow path.
            if TAKERATE_MEASUREMENT && self.mmtk.inside_harness() {
                FAST_COUNT.fetch_add(1, Ordering::SeqCst);
            }
            if let Ok(old) = self.log_slot_and_get_old_target(slot) {
                if TAKERATE_MEASUREMENT && self.mmtk.inside_harness() {
                    SLOW_COUNT.fetch_add(1, Ordering::SeqCst);
                }
                self.slow(src, slot, old);
                true
            } else {
                false
            }
        } else {
            // Non-heap slot (e.g. external GenericMemory data, how!=0).
            // No unlog bits exist for this address.
            //
            // This path is reached only from object_probable_write_slow
            // (internal GC scanning of all fields of a modified object),
            // NOT from the C write barrier.  The C barrier's
            // jl_gc_wb_field_pre uses mmtk_object_reference_write_pre_nonheap
            // for non-heap slots, which captures both old and new values
            // explicitly and pushes old → DEC_BUFFER, Direct(new) →
            // NONHEAP_INC_BUFFER — avoiding both the one-shot degradation
            // (HANDOFF §53) and the RC overcount from duplicate slot pushes.
            //
            // For object_probable_write_slow, the slot-based approach is
            // correct: this fires once per object per GC cycle (the parent's
            // first-field unlog bit deduplicates), so there is no overcount
            // from repeated pushes.  The old value is captured now (pre-
            // write); the slot is read at GC time (post-write) to get the
            // new value.  The malloc'd buffer remains valid because the
            // owning GenericMemory keeps it alive.
            //
            // ProcessIncs handles this slot correctly: the heap range guard
            // in unlog_and_load_rc_object skips the non-existent unlog bits;
            // record_mature_evac_remset skips (address_in_defrag → false);
            // store is valid (writable malloc'd memory, but nursery evac
            // is disabled so objects are never moved).
            let old = slot.load();
            self.slow(src, slot, old);
            true
        }
    }

    fn should_create_satb_packets(&self) -> bool {
        self.lxr.cm_enabled()
            && (self.lxr.cm_in_progress() || self.lxr.current_pause() == Some(Pause::FinalMark))
    }

    #[cold]
    fn flush_incs(&mut self) {
        if !self.incs.is_empty() {
            let incs = self.incs.take();
            self.lxr.rc.increase_inc_buffer_size(incs.len());
            self.mmtk.scheduler.work_buckets[WorkBucketStage::RCProcessIncs].add(ProcessIncs::<
                _,
                EDGE_KIND_MATURE,
            >::new(
                incs, self.lxr
            ));
        }
    }

    #[cold]
    fn flush_decs_and_satb(&mut self) {
        if !self.decs.is_empty() {
            let w = if self.should_create_satb_packets() {
                let decs = Arc::new(self.decs.take());
                self.mmtk.scheduler.work_buckets[WorkBucketStage::FinishConcurrentWork]
                    .add(ProcessModBufSATB::new_arc(decs.clone()));
                ProcessDecs::new_arc(decs, LazySweepingJobsCounter::new_decs())
            } else {
                let decs = self.decs.take();
                ProcessDecs::new(decs, LazySweepingJobsCounter::new_decs())
            };
            if crate::args::LAZY_DECREMENTS {
                self.mmtk.scheduler.postpone_prioritized(w);
            } else {
                self.mmtk.scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep].add(w);
            }
        }
    }

    #[cold]
    fn flush_weak_refs(&mut self) {
        if !self.refs.is_empty() {
            debug_assert!(self.should_create_satb_packets());
            let nodes = self.refs.take();
            self.mmtk.scheduler.work_buckets[WorkBucketStage::FinishConcurrentWork]
                .add(ProcessModBufSATB::new(nodes));
        }
    }
}

impl<VM: VMBinding> BarrierSemantics for LXRFieldBarrierSemantics<VM> {
    type VM = VM;

    #[cold]
    fn flush(&mut self) {
        self.flush_weak_refs();
        self.flush_incs();
        self.flush_decs_and_satb();
    }

    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        slot: VM::VMSlot,
        target: Option<ObjectReference>,
    ) {
        self.enqueue_node(Some(src), slot, target);
    }

    fn memory_region_copy_slow(&mut self, _src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        // Quickly check if all fields are logged. If yes, skip the barrier.
        let unlog_bits_start = address_to_meta_address(&Self::UNLOG_BITS, dst.start());
        let unlog_bits_start_aligned = unlog_bits_start.align_down(16);
        let unlog_bits_end =
            address_to_meta_address(&Self::UNLOG_BITS, dst.start() + dst.bytes() - 1);
        let unlog_bits_end_aligned = unlog_bits_end.align_down(16);
        let mut cursor = unlog_bits_start_aligned;
        let mut all_logged = true;
        while cursor <= unlog_bits_end_aligned {
            if unsafe { cursor.load::<u128>() } != 0 {
                all_logged = false;
                break;
            }
            cursor = cursor + 16usize;
        }
        if all_logged {
            return;
        }

        for s in dst.iter_slots() {
            let _succ = self.enqueue_node(None, s, None);
        }
    }

    fn load_weak_reference(&mut self, o: ObjectReference) {
        if !self.lxr.cm_in_progress() || self.lxr.is_marked(o) {
            return;
        }
        self.refs.push(o);
        if self.refs.is_full() {
            self.flush_weak_refs();
        }
    }

    fn object_probable_write_slow(&mut self, obj: ObjectReference) {
        obj.iterate_fields::<VM, _>(CLDScanPolicy::Ignore, RefScanPolicy::Follow, |s, _| {
            let _succ = self.enqueue_node(Some(obj), s, None);
        });
    }

    fn object_reference_write_post_cmpswap(
        &mut self,
        src: ObjectReference,
        slot: <Self::VM as VMBinding>::VMSlot,
        old: Option<ObjectReference>,
        _new: Option<ObjectReference>,
    ) {
        // Heap range check: skip slots outside the managed heap (no metadata).
        {
            use crate::util::heap::layout::vm_layout::vm_layout;
            let slot_addr = slot.to_address();
            let layout = vm_layout();
            if slot_addr < layout.heap_start || slot_addr >= layout.heap_end {
                return;
            }
        }
        // Atomically log the slot's unlog bit.  If already logged by another
        // thread or a prior barrier in this GC cycle, skip — the slot's old
        // value was already captured.
        if !self.attempt_to_log_field(slot) {
            return;
        }
        // Push explicit old → decs, slot → incs.
        // Unlike enqueue_node/log_slot_and_get_old_target, we do NOT read the
        // old value from the slot (which now contains the NEW value after the
        // successful cmpswap).  The caller provides the real old value from
        // the cmpswap's `expected` parameter.
        self.slow(Some(src), slot, old);
    }
}
