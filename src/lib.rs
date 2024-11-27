#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//! The slab-like allocator works using a method inspired by a SLAB allocator.
//! It uses regions of memory of `SLAB_SIZE` bytes divided up into object elements.
//! Each region (slab) is located using a self-contained doubly-linked list. The head contains a
//! pointer to the first and last slab, as a doubly linked list normally would.
//! However, it also maintains a cursor into a slab which will be used to allocate memory from next.
//!
//! Each slab is divided into object-elements (OE), and metadata. The metadata is allocated into the
//! first object elements within the slab. The metadata contains pointers to the previous and next
//! slabs and a free list.
//! The free list is a bitmap of OE, because the metadata struct is allocated to at
//! least one OE the first bit is used as a lock bit. To locate a free OE the leading ones
//! will be counted until a zero is found, which is set to one and the lock bit is cleared.
//!
//! If a slab is full the cursor will be moved to the next slab and which will be searched for a free OE.
//! Using this method full slabs will be moved "before" the cursor and empty slabs are after it reducing seek times.
//! If no free slabs are available then one will be allocated from the given allocator.
//!
//! The list can be sorted, this look back and move all slabs which aren't full ahead of the cursor.
//! Slabs ahead of the cursor are guaranteed to have free space so these do not need to be checked.

// TODO switch all the references to raw pointers

mod slab_meta;

use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use slab_meta::SlabMetadata;

pub use slab_meta::meta_bitmap_size;

pub struct SlabLike<A: core::alloc::Allocator, T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    alloc: A,

    // When a thread attempts to allocate memory it must increment the visitors counter.
    // If the allocation of the current cursor fails
    #[cfg(debug_assertions)]
    slab_count: core::sync::atomic::AtomicUsize,

    pointers: spin::RwLock<SlabLikePointers<T, SLAB_SIZE>>,
}

unsafe impl<A: Allocator, T, const SLAB_SIZE: usize> Send for SlabLike<A, T, SLAB_SIZE>
where
    A: Send,
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
}
unsafe impl<A: Allocator, T, const SLAB_SIZE: usize> Sync for SlabLike<A, T, SLAB_SIZE>
where
    A: Sync,
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
}

struct SlabLikePointers<T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    // Head and tail must always be the same variant
    head: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    tail: Option<NonNull<Slab<T, SLAB_SIZE>>>,
    cursor: Option<NonNull<Slab<T, SLAB_SIZE>>>,
}

impl<A: core::alloc::Allocator, T: 'static, const SLAB_SIZE: usize> SlabLike<A, T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub const fn new(alloc: A) -> Self {
        Self {
            alloc,
            #[cfg(debug_assertions)]
            slab_count: core::sync::atomic::AtomicUsize::new(0),
            pointers: spin::RwLock::new(SlabLikePointers {
                head: None,
                tail: None,
                cursor: None,
            }),
        }
    }

    fn new_slab(&self) -> Result<&'static mut Slab<T, SLAB_SIZE>, AllocError> {
        let ptr = self
            .alloc
            .allocate(Layout::from_size_align(size_of::<Slab<T, SLAB_SIZE>>(), SLAB_SIZE).unwrap());
        let slab_ptr = ptr?.cast();

        unsafe { slab_ptr.write(Slab::<T, SLAB_SIZE>::new()) };
        let n_slab: &mut Slab<T, SLAB_SIZE> = unsafe { &mut *ptr?.as_ptr().cast() };
        #[cfg(debug_assertions)]
        self.slab_count
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        Ok(n_slab)
    }

    pub fn count_slabs(&self) -> usize {
        let lr = self.pointers.read();
        let mut slab = if let Some(n) = lr.head {
            unsafe { &*n.as_ptr() }
        } else {
            debug_assert!(lr.tail.is_none());
            debug_assert!(lr.cursor.is_none());
            return 0;
        };

        let mut count = 1;
        loop {
            match slab.next_slab() {
                Some(p) => slab = p,
                None => return count,
            }
            count = count.checked_add(1).expect("slab count exceeds usize::MAX");
        }
    }

    pub fn sanitize(&self) {
        self.sanitize_inner();
    }

    fn sanitize_inner(&self) -> Option<()> {
        let mut wl = self.pointers.write();
        //let mut slab = wl.cursor.map(|p| p.as_ptr());
        let mut iter = wl.iter(false);
        while iter.not_exhausted() {
            // hitting ? here indicates that all slabs are full
            let (split_start, split_off_tail) = iter.find_split_tail()?;
            let mut split_off_tail = core::mem::MaybeUninit::new(split_off_tail);
            #[cfg(all(test, feature = "debug-print"))]
            {
                eprintln!("split_start: {split_start:p}");
            }
            #[cfg(all(test, feature = "debug-print"))]
            {
                unsafe {
                    eprintln!("split_off_tail: {:p}", split_off_tail.assume_init_read());
                }
            }

            let (split_off_head, mut split_end) = iter.find_split_head();
            let mut split_off_head = core::mem::MaybeUninit::new(split_off_head); // this might be the same as `split_off_start` so it is treated as volatile
            #[cfg(all(test, feature = "debug-print"))]
            {
                unsafe {
                    eprintln!("split_off_head: {:p}", split_off_head.assume_init_read());
                }
            }
            #[cfg(all(test, feature = "debug-print"))]
            {
                if let Some(se) = unsafe { core::ptr::read(&split_end) } {
                    eprintln!("split_end: {se:p}");
                }
            }

            #[cfg(all(test, feature = "debug-print"))]
            eprintln!("-------------");

            // split ends are joined up.
            if let Some(ref mut end) = split_end {
                end.set_next(Some(split_start));
            }
            if split_end.is_none() {
                // SAFETY: ref cannot be non-null
                wl.head = Some(unsafe { NonNull::new_unchecked(split_start) });
            }
            split_start.set_prev(split_end);

            // SAFETY: We need to prevent the compiler from making assumptions
            // both split_off*'s are initialized
            unsafe { split_off_tail.assume_init_mut().set_next(None) };
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst); // I might be overthinking this
                                                                                      // link split head to the tail
            let sp_head = unsafe { split_off_head.assume_init_mut() };
            let tail = unsafe { &mut *wl.tail.as_ref().unwrap().as_ptr() };
            tail.set_next(Some(sp_head));
            sp_head.set_prev(Some(tail));
            // SAFETY: This is safe this is guaranteed to be valid its wrapped in MaybeUninit to prevent the compiler manhandling it.
            wl.tail = Some(NonNull::new(unsafe { split_off_tail.assume_init_read() }).unwrap());
        }

        Some(())
    }

    /// Extends allocator to fit `size` more objects.
    pub fn extend(&self, size: usize) -> Result<(), AllocError> {
        let new_slabs = size.div_ceil(slab_count_obj_elements::<T, SLAB_SIZE>());
        let mut wl = self.pointers.write();
        // It might be better to allocate all new slabs into a vec to reduce how long we have the write lock.
        // we can also optimize this by switching it to a proper linked list insertion.
        for _ in 0..new_slabs {
            let n = self.new_slab()?;
            // SAFETY: n is new.
            unsafe { wl.new_append(n) }
        }

        Ok(())
    }

    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    fn sanity_check(&self, maybe_empty: bool) {
        let lr = self.pointers.read();
        let mut slab = if let Some(n) = lr.head {
            unsafe { &*n.as_ptr() }
        } else {
            assert!(lr.tail.is_none());
            assert!(lr.cursor.is_none());
            assert!(maybe_empty, "List was empty when it was not expected");
            return;
        };

        let mut count = 1;

        let mut hit_cursor =
            core::ptr::addr_eq(slab, lr.cursor.expect("Cursor must be set").as_ptr());

        loop {
            match slab.next_slab() {
                Some(p) => {
                    if core::ptr::addr_eq(p, lr.cursor.expect("Cursor must be set").as_ptr()) {
                        hit_cursor = true;
                    }
                    count += 1;
                    slab = p
                }
                None => break,
            }
        }

        assert_eq!(
            count,
            self.slab_count.load(core::sync::atomic::Ordering::Relaxed),
            "Incorrect number of slabs"
        );
        assert!(hit_cursor);
        assert!(core::ptr::addr_eq(lr.tail.unwrap().as_ptr(), slab))
    }

    /// Attempts to remove `shrink` slabs from the linked list.
    /// If `count` slabs are removed then this will return `Ok(count)` else it will return
    /// `Err(n)` where `n` is the number of slabs removed.
    ///
    /// This will only scan slabs after the cursor. The caller should call [Self::sanitize] before
    /// calling this to scan all available slabs.
    ///
    /// This fn will return `Err(0)` if the cursor is not set.
    ///
    /// This method blocks allocation.
    #[allow(dead_code)]
    pub fn free_slabs(&self, shrink: usize) -> Result<usize, usize> {
        let mut wl = self.pointers.write();

        let mut count = 0;

        for i in wl.iter(true) {
            if i.is_empty() {
                let mut last = i.prev_slab();
                let mut next = i.next_slab();
                if let Some(ref mut l) = last {
                    l.set_next(i.next_slab());
                }
                if let Some(ref mut n) = next {
                    n.set_prev(i.prev_slab());
                }

                macro_rules! rm_ptr {
                    ($ptr:expr, $slab:ident, $replace:expr) => {
                        match $ptr {
                            Some(cursor) if core::ptr::addr_eq(cursor.as_ptr(), $slab) => {
                                $ptr = $replace.map(|p| unsafe { NonNull::new_unchecked(p) });
                            }
                            _ => {}
                        }
                    };
                }

                rm_ptr!(wl.cursor, i, i.next_slab());
                rm_ptr!(wl.head, i, i.next_slab());
                rm_ptr!(wl.tail, i, i.prev_slab());

                unsafe {
                    self.alloc.deallocate(
                        NonNull::new_unchecked(i).cast(),
                        Layout::from_size_align(size_of::<Slab<T, SLAB_SIZE>>(), SLAB_SIZE)
                            .unwrap(),
                    )
                };
                count += 1;
                #[cfg(debug_assertions)]
                self.slab_count
                    .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
                if count >= shrink {
                    break;
                }
            }
        }

        if count == shrink {
            Ok(count)
        } else {
            Err(count)
        }
    }

    #[cfg(test)]
    fn info_dump(&self) {
        let wl = self.pointers.write();
        let iter = SlabCursor {
            next: wl.head.map(|p| unsafe { &mut *p.as_ptr() }),
            forward: true,
        };

        for (n, i) in iter.enumerate() {
            eprint!("{n}: {:?}, {i:p}", i.slab_metadata.bitmap());
            if core::ptr::addr_eq(i, wl.cursor.unwrap().as_ptr()) {
                eprint!("Cursor");
            }
            eprintln!()
        }
    }
}

impl<T: 'static, const SLAB_SIZE: usize> SlabLikePointers<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    /// Attempts to allocate from the cursor. On failure the cursor must be advanced.
    fn allocate(&self) -> Result<NonNull<[u8]>, AllocError> {
        let slab = self
            .cursor
            .map(|cursor| unsafe { &mut *cursor.as_ptr() })
            .ok_or(AllocError)?;
        slab.alloc()
    }

    /// Moves the cursor to the next slab in the list.
    ///
    /// When this returns `None` a new slab must be allocated.
    fn advance_cursor(&mut self) -> Result<(), ()> {
        let slab = unsafe { &mut *self.cursor.ok_or(())?.as_ptr() };
        if let Some(slab) = slab.next_slab() {
            // SAFETY: References cannot be null
            self.cursor = unsafe { Some(NonNull::new_unchecked(slab)) };
            Ok(())
        } else {
            Err(())
        }
    }

    /// Allocates a new slab, appends it to the end of the linked list.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `slab` is not already within the list.
    unsafe fn new_append(&mut self, slab: &'static mut Slab<T, SLAB_SIZE>) {
        if self.head.is_none() {
            // Only slab, initialize all pointers
            self.head = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
            self.tail = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
            self.cursor = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
        } else {
            let tail = unsafe { &mut *self.tail.expect("Tail not initialized").as_ptr() };
            let _r = tail.set_next(Some(slab));
            slab.set_prev(Some(tail));
            debug_assert!(_r.is_none());
            self.tail = Some(NonNull::new(slab as *mut Slab<T, SLAB_SIZE>).unwrap());
        }
    }

    /// Returns an iterator over slabs starting at `self.cursor`. If `forward` is true then the
    /// iter will yield elements ahead of the cursor else elements behind the cursor will be yielded.
    ///
    /// If `self.cursor` is not set then this will yield [None]
    fn iter(&mut self, forward: bool) -> SlabCursor<T, SLAB_SIZE> {
        SlabCursor {
            next: self.cursor.map(|p| unsafe { &mut *p.as_ptr() }),
            forward,
        }
    }
}

unsafe impl<A: core::alloc::Allocator, T: 'static, const SLAB_SIZE: usize> core::alloc::Allocator
    for SlabLike<A, T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert_eq!(
            layout,
            Layout::new::<T>(),
            "Unexpected layout for {}: {layout:?}, expected {:?}",
            core::any::type_name::<Self>(),
            Layout::new::<T>()
        );

        loop {
            let rl = self.pointers.read();
            if let Ok(ptr) = rl.allocate() {
                return Ok(ptr);
            } else {
                // Only one thread should update the slab.
                // This blocks further calls to read().
                if let Some(ur) = self.pointers.try_upgradeable_read() {
                    // Drop rl to prevent blocking while upgrading.
                    // Dropping it after acquiring upgradeable prevents housekeeping tasks from running in this time.
                    drop(rl);
                    // Waits until all threads drop rl
                    let mut wl = ur.upgrade();
                    if wl.advance_cursor().is_err() {
                        // SAFETY: The slab is new and cannot be within the list.
                        // Can this be optimized to allocate the new slab before ur.upgrade()?
                        unsafe { wl.new_append(self.new_slab()?) }
                    }
                }
            }
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        debug_assert_eq!(
            layout,
            Layout::new::<T>(),
            "Unexpected layout for {}: {layout:?}, expected {:?}",
            core::any::type_name::<Self>(),
            Layout::new::<T>()
        );

        // This takes `ptr` aligns it down to `SLAB_SIZE` and casts it to Slab
        // Because `Self` forces alignment of Slabs to `SLAB_SIZE` we can guarantee that this is indeed the pointer into a Slab.
        let tgt_slab = unsafe {
            &mut *(((ptr.as_ptr() as usize) & !(SLAB_SIZE - 1)) as *mut Slab<T, SLAB_SIZE>)
        };

        // offset into array
        let offset = (ptr.as_ptr() as usize) - ((&tgt_slab.obj_elements) as *const _ as usize);
        let index = if offset == 0 {
            0
        } else {
            // offset divided by element size, returns element index
            offset / ((tgt_slab.obj_elements.len()) / size_of_val(&tgt_slab.obj_elements))
        };

        tgt_slab.free(index)
    }
}

/// Returns the number of object elements that can be stored in a [Slab]
///
/// This fn will panic if the number of object elements is 0.
pub const fn slab_count_obj_elements<T, const SLAB_SIZE: usize>() -> usize {
    let elem_size = size_of::<[T; 2]>() / 2;
    let raw_obj_elem_count = SLAB_SIZE / elem_size;

    let meta_size = slab_meta::size_of_meta::<T>(SLAB_SIZE);
    let meta_obj_elem_size = meta_size.div_ceil(elem_size);

    let num_obj_elements = raw_obj_elem_count - meta_obj_elem_size;
    assert!(
        num_obj_elements > 0,
        "Slab must have more than one Object elements"
    ); // This'd be a shit allocator if we had 100% overhead

    num_obj_elements
}

#[repr(C)]
struct Slab<T, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    slab_metadata: SlabMetadata<T, SLAB_SIZE>,
    obj_elements: [core::mem::MaybeUninit<T>; slab_count_obj_elements::<T, SLAB_SIZE>()],
}

impl<T: 'static, const SLAB_SIZE: usize> Slab<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    pub fn new() -> Self
    where
        [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
        [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
    {
        Self {
            slab_metadata: SlabMetadata::new(),
            obj_elements: [const { core::mem::MaybeUninit::uninit() };
                slab_count_obj_elements::<T, SLAB_SIZE>()],
        }
    }

    fn alloc(&mut self) -> Result<NonNull<[u8]>, AllocError> {
        let index = self.slab_metadata.alloc().ok_or(AllocError)?
            - SlabMetadata::<T, SLAB_SIZE>::reserved_bits();

        // uses get() because the index may be out of bounds. We check this here because im sure the
        // compiler will bitch if SlabMetadata does it.
        let u =
            self.obj_elements.get_mut(index).ok_or(AllocError)? as *mut core::mem::MaybeUninit<T>;
        // SAFETY: Pointer is taken from a reference
        Ok(unsafe {
            NonNull::new_unchecked(core::slice::from_raw_parts_mut(u.cast(), size_of::<T>()))
        })
    }

    fn free(&self, index: usize) {
        let bit = slab_meta::SlabMetadata::<T, SLAB_SIZE>::reserved_bits() + index;
        self.slab_metadata.free(bit);
    }

    fn next_slab(&self) -> Option<&'static mut Self> {
        Some(unsafe { &mut *self.slab_metadata.next_slab()? })
    }

    fn prev_slab(&self) -> Option<&'static mut Self> {
        Some(unsafe { &mut *self.slab_metadata.prev_slab()? })
    }

    fn set_next(&self, slab: Option<&mut Self>) -> Option<&'static mut Self> {
        self.slab_metadata
            .set_next(slab.map(|p| p as *mut _))
            .map(|p| unsafe { &mut *p })
    }
    fn set_prev(&self, slab: Option<&mut Self>) -> Option<&'static mut Self> {
        self.slab_metadata
            .set_prev(slab.map(|p| p as *mut _))
            .map(|p| unsafe { &mut *p })
    }

    /// Returns the number of free object elements in `self`
    fn query_free(&self) -> usize {
        slab_count_obj_elements::<T, SLAB_SIZE>() - self.slab_metadata.allocated()
    }

    fn is_empty(&self) -> bool {
        self.slab_metadata.allocated() == 0
    }
}

struct SlabCursor<T: 'static, const SLAB_SIZE: usize>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    next: Option<&'static mut Slab<T, SLAB_SIZE>>,
    forward: bool,
}

impl<T, const SLAB_SIZE: usize> SlabCursor<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    fn not_exhausted(&self) -> bool {
        self.next.is_some()
    }

    /// Attempts to locate a slab with free space.
    /// If one is found it will return the slab before the empty slab and the empty slab `Some(found_first,empty_slab)`
    fn find_split_tail(
        &mut self,
    ) -> Option<(
        &'static mut Slab<T, SLAB_SIZE>,
        &'static mut Slab<T, SLAB_SIZE>,
    )> {
        while let Some(slab) = self.next() {
            if self.forward && slab.next_slab().is_some_and(|s| s.query_free() != 0) {
                let ns = slab.next_slab().unwrap();
                return Some((slab, ns));
            } else if slab.prev_slab().is_some_and(|s| s.query_free() != 0) {
                let ns = slab.prev_slab().unwrap();
                return Some((slab, ns));
            }
        }

        None
    }

    /// Attempts to locate a slab with free space, when one is found the slab before is yielded.
    /// If no slab is found then this will return the last slab in the list.
    /// The caller should call [Self::peek] after calling this to get the next slab.
    fn find_split_head(
        &mut self,
    ) -> (
        &'static mut Slab<T, SLAB_SIZE>,
        Option<&'static mut Slab<T, SLAB_SIZE>>,
    ) {
        while let Some(slab) = self.next() {
            if self.forward && slab.next_slab().is_some_and(|s| s.query_free() == 0) {
                let ns = slab.next_slab();
                return (slab, ns);
            } else if slab.prev_slab().is_some_and(|s| s.query_free() == 0) {
                let ns = slab.prev_slab();
                return (slab, ns);
            } else if !self.not_exhausted() {
                // will return true when the final slab is yielded.
                return (slab, None); // this is the last slab in the list
            }
        }
        unreachable!()
    }
}

impl<T: 'static, const SLAB_SIZE: usize> Iterator for SlabCursor<T, SLAB_SIZE>
where
    [(); meta_bitmap_size::<T>(SLAB_SIZE)]:,
    [(); slab_count_obj_elements::<T, SLAB_SIZE>()]:,
{
    type Item = &'static mut Slab<T, SLAB_SIZE>;
    fn next(&mut self) -> Option<Self::Item> {
        let s = self.next.take()?;
        if self.forward {
            self.next = s.next_slab();
        } else {
            self.next = s.prev_slab();
        }
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::boxed::Box;

    #[test]
    fn check_slab_obj_count() {
        let slab = Slab::<u16, 64>::new();
        //assert_eq!(core::mem::size_of::<Slab<u128, 64>>(), 64);
        assert_eq!(slab.obj_elements.len(), 16);
        let slab = Slab::<u128, 64>::new();
        assert_eq!(core::mem::size_of::<Slab<u128, 64>>(), 64);
        assert_eq!(slab.obj_elements.len(), 2);
    }

    #[test]
    fn check_slab_geom_size() {
        assert_eq!(size_of::<Slab<u8, 64>>(), 64);
        assert_eq!(size_of::<Slab<u8, 128>>(), 128);
        assert_eq!(size_of::<Slab<u128, 128>>(), 128);
        assert_eq!(size_of::<Slab<u64, 4096>>(), 4096);
    }

    #[test]
    fn obj_elem_count() {
        assert_eq!(slab_count_obj_elements::<u128, 64>(), 2);
        assert_eq!(slab_count_obj_elements::<u16, 64>(), 16);
    }

    #[test]
    fn test_slab_alloc() {
        let mut slab = Slab::<u128, 64>::new();
        let slab_addr = &slab as *const _ as usize;
        let first = slab.alloc().unwrap();
        assert_eq!(
            first.cast::<u8>().as_ptr() as usize,
            slab_addr + 32,
            "base: {slab_addr:x}"
        );
        let second = slab.alloc().unwrap();
        assert_eq!(
            second.cast::<u8>().as_ptr() as usize,
            slab_addr + 48,
            "base: {slab_addr:x}"
        );
    }

    #[test]
    fn test_slab_alloc_u16() {
        let mut slab = Slab::<u16, 4096>::new();
        let slab_addr = &slab as *const _ as usize;
        let base = &slab as *const _ as usize;
        for i in 0..1776 {
            // 1776 is expected number of obj-elements
            let ptr = slab.alloc().unwrap().cast::<u8>().as_ptr() as usize;
            assert_eq!(
                ptr,
                base + 280 + (2 * i),
                "Iteration {i}, base {base:#x}, meta_end {:#x}",
                base + 272
            );
        }
    }

    #[test]
    fn init_slablike() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u16, 4096>::new(Global);

        let _b = Box::try_new_in(0u16, &sl).unwrap();

        let sl = SlabLike::<Global, u128, 64>::new(Global);

        let _b = Box::try_new_in(0u128, &sl).unwrap();
    }

    #[test]
    fn slablike_free() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u16, 4096>::new(Global);

        let b = Box::try_new_in(0u16, &sl).unwrap();
        let addr = &*b as *const _ as usize;
        drop(b);
        let b = Box::try_new_in(0u16, &sl).unwrap();
        assert_eq!(&*b as *const _ as usize, addr);
    }

    #[test]
    fn slablike_auto_extend() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u128, 64>::new(Global);

        for i in 0..8 {
            let b = Box::try_new_in(0u128, &sl).expect(&std::format!("Failed on iteration {i}"));
            let _ = Box::leak(b);
        }
    }

    #[test]
    fn slablike_go_hard() {
        use std::alloc::Global;
        let sl = SlabLike::<Global, u8, 64>::new(Global);

        for i in 0..0x100_0000 {
            let b = Box::new_in(0u8, &sl);
        }
    }

    #[test]
    fn slablike_concurrent() {
        use std::alloc::Global;
        const SL: SlabLike<Global, u8, 64> = SlabLike::<Global, u8, 64>::new(Global);
        let mut threads = std::vec::Vec::with_capacity(16);

        for i in 0..16 {
            threads.push(std::thread::spawn(move || {
                let tid = std::thread::current().id();
                for n in 0..0x10_0000 {
                    let mut count: u8 = 0;
                    let mut b = Box::new_in(count, SL);

                    std::hint::spin_loop();
                    assert_eq!(*b, count);
                    count = count.wrapping_add(1);
                    b = Box::new_in(count, SL);
                }
            }));
        }

        for i in threads {
            i.join().unwrap();
        }
    }

    #[test]
    fn sizing_and_extending() {
        let sl = SlabLike::<std::alloc::Global, u8, 64>::new(std::alloc::Global);
        sl.extend(128).unwrap();
        eprintln!("{}", size_of::<SlabMetadata::<u8, 64>>());

        assert_eq!(sl.count_slabs(), 4)
    }

    #[test]
    fn test_housekeeping() {
        use rand::prelude::*;
        static SL: SlabLike<std::alloc::Global, u8, 64> = SlabLike::new(std::alloc::Global);
        let sl = &SL;

        const ALLOC_COUNT: usize = 0x1_0000;

        let mut rng = thread_rng();
        let mut buff = std::vec::Vec::new();
        for _ in 0..ALLOC_COUNT {
            buff.push(std::boxed::Box::new_in(0u8, &SL));
        }

        let p = &*buff;

        for _ in 0..32 {
            buff.shuffle(&mut rng);
            std::thread::sleep(std::time::Duration::from_millis(10));
            for _ in 0..ALLOC_COUNT / 2 {
                assert_eq!(*buff.pop().unwrap(), 0)
            }

            SL.sanity_check(false);
            SL.sanitize();
            SL.sanity_check(false);

            for count in 0..ALLOC_COUNT / 2 {
                buff.push(std::boxed::Box::new_in(0, &SL));
            }
            SL.sanity_check(false);
        }
    }

    #[test]
    fn housekeeping_noisy() {
        use rand::prelude::*;
        static SL: SlabLike<std::alloc::Global, u8, 64> = SlabLike::new(std::alloc::Global);

        //let mut rand = StdRng::seed_from_u64(42069);
        let mut rand = thread_rng();
        let mut buff = std::vec::Vec::new();
        for _ in 0..128 {
            let chunk: [std::boxed::Box<u8, &SlabLike<std::alloc::Global, u8, 64>>; 32] =
                core::array::from_fn(|_| std::boxed::Box::new_in(0u8, &SL));
            buff.push(chunk);
        }

        buff.shuffle(&mut rand);

        for _ in 0..32 {
            buff.pop();
        }
        {
            let mut wl = SL.pointers.write();
            let it = SlabCursor {
                next: wl.head.map(|p| unsafe { &mut *p.as_ptr() }),
                forward: true,
            };
            for (i, s) in it.enumerate() {
                eprint!("{i}: {:?} {s:p}", s.slab_metadata.bitmap());
                if core::ptr::addr_eq(s, wl.cursor.unwrap().as_ptr()) {
                    eprintln!(" Cursor");
                } else {
                    eprintln!();
                }
            }
        }

        eprintln!("Clean");
        SL.sanitize();
        SL.sanity_check(false);

        {
            let mut wl = SL.pointers.write();
            let it = SlabCursor {
                next: wl.head.map(|p| unsafe { &mut *p.as_ptr() }),
                forward: true,
            };
            for (i, s) in it.enumerate() {
                eprint!("{i}: {:?} {s:p}", s.slab_metadata.bitmap());
                if core::ptr::addr_eq(s, wl.cursor.unwrap().as_ptr()) {
                    eprintln!(" Cursor");
                } else {
                    eprintln!();
                }
            }

            // Skip cursor because it may be full/partial/empty
            for i in wl.iter(true).skip(1) {
                assert!(i.query_free() > 0, "Found full slab ahead of cursor");
            }
            for (n, i) in wl.iter(false).skip(1).enumerate() {
                eprintln!("{i:p}");
                assert_eq!(
                    i.query_free(),
                    0,
                    "Found partial slab behind of cursor: n:{n}: {i:p}"
                );
            }
        }
    }

    #[test]
    fn check_housekeeping_concurrent() {
        extern crate std;
        use rand::prelude::*;
        const TEST_SIZE: usize = 1_0000;
        static SL: SlabLike<std::alloc::Global, u8, 16384> = SlabLike::new(std::alloc::Global);
        let mut threads = std::vec::Vec::new();
        for thread in 0..20u8 {
            threads.push(std::thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(69);
                let mut buff = std::vec::Vec::new();
                for _ in 0..TEST_SIZE {
                    buff.push(std::boxed::Box::try_new_in(thread, &SL).unwrap());
                }

                let p = &*buff;

                for _ in 0..32 {
                    buff.shuffle(&mut rng);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    for _ in 0..TEST_SIZE / 2 {
                        assert_eq!(*buff.pop().unwrap(), thread)
                    }
                    SL.sanitize();
                    for _ in 0..TEST_SIZE / 2 {
                        buff.push(std::boxed::Box::try_new_in(thread, &SL).unwrap());
                    }
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }
        SL.sanity_check(false);
    }

    #[test]
    fn check_shrink_simple() {
        static SL: SlabLike<std::alloc::Global, u8, 64> = SlabLike::new(std::alloc::Global);

        SL.extend(256).unwrap();
        SL.sanitize();
        SL.sanity_check(false);
        assert_eq!(SL.free_slabs(8), Ok(8), "All slabs should've been removed");
        SL.sanity_check(true);
        SL.extend(256).unwrap();
        SL.sanitize();
        let b = std::boxed::Box::new_in(0u8, &SL);
        SL.info_dump();
        assert_eq!(SL.free_slabs(8), Err(7), "All slabs should've been removed");
        SL.sanity_check(false);
        drop(b);
        SL.sanitize();
        assert_eq!(SL.free_slabs(1), Ok(1), "All slabs should've been removed");
    }

    #[test]
    fn check_shrink_random() {
        use rand::prelude::*;
        static SL: SlabLike<std::alloc::Global, u8, 64> = SlabLike::new(std::alloc::Global);
        let mut rand = StdRng::seed_from_u64(69);

        let mut buff = std::vec::Vec::new();
        for _ in 0..128 {
            let chunk: [std::boxed::Box<u8, &SlabLike<std::alloc::Global, u8, 64>>; 32] =
                core::array::from_fn(|_| std::boxed::Box::new_in(0u8, &SL));
            buff.push(chunk);
        }
        buff.shuffle(&mut rand);

        for _ in 0..64 {
            buff.pop();
        }
        SL.sanitize();

        assert_eq!(SL.free_slabs(usize::MAX), Err(64))
    }

    #[test]
    fn check_shrink_rand_with_noise() {
        use rand::prelude::*;
        static SL: SlabLike<std::alloc::Global, u8, 64> = SlabLike::new(std::alloc::Global);
        let mut rand = rand::thread_rng();

        let mut buff = std::vec::Vec::new();
        for _ in 0..128 {
            let chunk: [std::boxed::Box<u8, &SlabLike<std::alloc::Global, u8, 64>>; 32] =
                core::array::from_fn(|_| std::boxed::Box::new_in(0u8, &SL));
            buff.push(chunk);
        }
        buff.shuffle(&mut rand);

        for _ in 0..32 {
            buff.pop();
        }

        let mut flat_buff = buff.into_iter().flatten().collect::<Vec<_>>();
        flat_buff.shuffle(&mut rand);
        for _ in 0..1536 {
            // that's half of the remaining size
            flat_buff.pop();
        }

        SL.sanitize();
        assert_eq!(SL.free_slabs(32), Ok(32));
        assert_eq!(
            SL.free_slabs(usize::MAX),
            Err(0),
            "This may spuriously fail due to randomness, rerun this a few times"
        );
    }
}
