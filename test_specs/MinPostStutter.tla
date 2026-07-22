---- MODULE MinPostStutter ----
(* More faithful reproduction of PostStutter for #147 *)
EXTENDS Naturals

VARIABLES s, pc, h_turn

top == [top |-> "top"]
Procs == {1, 2}
Other(self) == IF self = 1 THEN 2 ELSE 1

(* Original action without stuttering *)
l1(self) == /\ pc[self] = "l1"
            /\ pc' = [pc EXCEPT ![self] = "cs"]

(* PostStutter-style wrapping *)
PostStutter(A, actionId, context, bot, initVal, decr(_)) ==
  IF s = top
  THEN /\ A
       /\ s' = [id |-> actionId, ctxt |-> context, val |-> initVal]
  ELSE /\ s.id = actionId
       /\ s.ctxt = context
       /\ UNCHANGED <<pc>>
       /\ s'= IF s.val = bot THEN top
                             ELSE [s EXCEPT !.val = decr(s.val)]

(* LockHS-style action wrapping *)
l1HS(self) ==
  /\ PostStutter(l1(self), "l1", self, 1, 2, LAMBDA j : j-1)
  /\ h_turn' = IF s' # top THEN IF s'.val = 1 THEN Other(self)
                                             ELSE h_turn
                          ELSE h_turn

Init == /\ pc = [p \in Procs |-> "l1"]
        /\ s = top
        /\ h_turn = 1

Next == \E self \in Procs : l1HS(self)
====
