use core::ops::Deref;

pub struct GateKeeper {
    inner: spin::Mutex<GateKeeperInner>,
}
struct GateKeeperInner {
    ticket: usize,
    concurrent: usize,
}

impl GateKeeper {
    pub const fn new() -> Self {
        Self {
            inner: spin::Mutex::new(GateKeeperInner {
                ticket: 0,
                concurrent: 0,
            }),
        }
    }

    pub fn get(&self) -> Ticket {
        let mut l = self.inner.lock();
        let ticket = l.ticket;
        l.concurrent += 1;
        l.ticket += 1;

        Ticket {
            gatekeeper: self,
            ticket,
        }
    }
}

pub struct Ticket<'a> {
    gatekeeper: &'a GateKeeper,
    ticket: usize,
}

impl Deref for Ticket<'_> {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.ticket
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        let mut l = self.gatekeeper.inner.lock();
        l.concurrent -= 1;
        if l.concurrent == 0 {
            l.ticket = 0;
        }
    }
}
