------------------------------ MODULE Option -----------------------------------
(*
 * Option type operators for Apalache.
 *
 * An Option is a Variant with tag "Some" or "None":
 *   Some(val)  == Variant("Some", val)
 *   None       == Variant("None", UNIT)
 *
 * Reference: https://apalache-mc.org/docs/lang/apalache-operators.html
 *
 * The bodies below are TLC-compatible fallbacks. TY provides native Rust
 * builtin implementations for performance where applicable.
 *)

EXTENDS Apalache, Variants

(**
 * Wrap a value in a Some variant.
 * @type: a => [tag: Str, value: a];
 *)
Some(__val) == Variant("Some", __val)

(**
 * The None variant.
 * @type: [tag: Str, value: Str];
 *)
None == Variant("None", UNIT)

(**
 * Check if an option is Some.
 * @type: [tag: Str, value: a] => Bool;
 *)
IsSome(__opt) == VariantTag(__opt) = "Some"

(**
 * Check if an option is None.
 * @type: [tag: Str, value: a] => Bool;
 *)
IsNone(__opt) == VariantTag(__opt) = "None"

(**
 * Pattern match on an option.
 * @type: ([tag: Str, value: a], (a => b), (() => b)) => b;
 *)
OptionCase(__opt, __someFn, __noneFn) ==
    IF IsSome(__opt)
    THEN __someFn(VariantGetUnsafe("Some", __opt))
    ELSE __noneFn(UNIT)

(**
 * Map a function over the Some value.
 * @type: ([tag: Str, value: a], (a => b)) => [tag: Str, value: b];
 *)
OptionMap(__opt, __fn) ==
    IF IsSome(__opt)
    THEN Some(__fn(VariantGetUnsafe("Some", __opt)))
    ELSE None

(**
 * FlatMap a function over the Some value.
 * @type: ([tag: Str, value: a], (a => [tag: Str, value: b])) => [tag: Str, value: b];
 *)
OptionFlatMap(__opt, __fn) ==
    IF IsSome(__opt)
    THEN __fn(VariantGetUnsafe("Some", __opt))
    ELSE None

(**
 * Extract the Some value or return a default.
 * @type: ([tag: Str, value: a], a) => a;
 *)
OptionGetOrElse(__opt, __default) ==
    VariantGetOrElse("Some", __opt, __default)

(**
 * Convert an option to a sequence.
 * @type: [tag: Str, value: a] => Seq(a);
 *)
OptionToSeq(__opt) ==
    IF IsSome(__opt)
    THEN <<VariantGetUnsafe("Some", __opt)>>
    ELSE <<>>

(**
 * Convert an option to a set.
 * @type: [tag: Str, value: a] => Set(a);
 *)
OptionToSet(__opt) ==
    IF IsSome(__opt)
    THEN {VariantGetUnsafe("Some", __opt)}
    ELSE {}

(**
 * Pick an element from a set, returning Some if non-empty, None otherwise.
 * @type: Set(a) => [tag: Str, value: a];
 *)
OptionGuess(__set) ==
    IF __set = {}
    THEN None
    ELSE Some(Guess(__set))

(**
 * Safe function application: returns Some(f[key]) if key is in DOMAIN f, None otherwise.
 * @type: ((a -> b), a) => [tag: Str, value: b];
 *)
OptionFunApp(__fn, __key) ==
    IF __key \in DOMAIN __fn
    THEN Some(__fn[__key])
    ELSE None

(**
 * Partial function wrapper: restrict function domain and wrap in Option.
 * @type: ((a -> b), Set(a)) => (a -> [tag: Str, value: b]);
 *)
OptionPartialFun(__fn, __dom) ==
    [__x \in DOMAIN __fn |->
        IF __x \in __dom
        THEN Some(__fn[__x])
        ELSE None]

================================================================================
