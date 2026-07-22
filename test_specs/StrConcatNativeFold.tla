---- MODULE StrConcatNativeFold ----
\* Cross-backend soundness fixture for native string `\o` (StrConcat) folding.
\*
\* `Build` sets the string state variable `label` to the result of `\o` applied
\* to two compile-time-known string constants ("key_" \o "42" = "key_42"). The
\* trust-ir lowering folds this `Concat` natively to the interned NameId of
\* "key_42" (see `lower_string_concat_const`), bit-identical to the bytecode
\* VM's `Value::string("key_42")`. The invariant `LabelNeverKey42` is genuinely
\* VIOLATED at the reachable phase-1 state, so the interpreter and the
\* trust-codegen native backend must agree on the violation.

VARIABLES phase, label

Init ==
  /\ phase = 0
  /\ label = "init"

\* `\o` on two string constants: the only soundly foldable shape for the
\* native lowering. Produces "key_42".
Build ==
  /\ phase = 0
  /\ phase' = 1
  /\ label' = "key_" \o "42"

Done ==
  /\ phase = 1
  /\ phase' = 1
  /\ label' = label

Next ==
  \/ Build
  \/ Done

Spec == Init /\ [][Next]_<<phase, label>>

TypeOK ==
  /\ phase \in {0, 1}
  /\ label \in {"init", "key_42"}

\* VIOLATED once `Build` fires and `label` becomes "key_42".
LabelNeverKey42 == label # "key_42"

====
