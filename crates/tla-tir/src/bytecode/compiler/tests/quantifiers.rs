// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[test]
fn diag_forall_tuple_pattern() {
    // \A <<a, b>> \in { <<1,2>> } : a + b >= 0
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Forall {
        vars: vec![TirBoundVar {
            name: "<<a, b>>".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![spanned(
                TirExpr::Tuple(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ]),
            )])))),
            pattern: Some(TirBoundPattern::Tuple(vec![
                ("a".to_string(), tla_core::NameId(0)),
                ("b".to_string(), tla_core::NameId(0)),
            ])),
        }],
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::ArithBinOp {
                left: Box::new(spanned(TirExpr::Name(ident_name("a")))),
                op: TirArithOp::Add,
                right: Box::new(spanned(TirExpr::Name(ident_name("b")))),
            })),
            op: TirCmpOp::Gt,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(0),
                ty: TirType::Int,
            })),
        })),
    });
    compiler
        .compile_expression("test", &expr)
        .expect("tuple-pattern Forall should compile by destructuring the binder");
}

#[test]
fn test_compile_choose() {
    // CHOOSE x \in {1, 2, 3} : x > 1
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirCmpOp::Gt,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(1),
                ty: TirType::Int,
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("CHOOSE should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let has_choose_begin = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseBegin { .. }));
    let has_choose_next = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseNext { .. }));
    assert!(has_choose_begin, "should emit ChooseBegin");
    assert!(has_choose_next, "should emit ChooseNext");
}

#[test]
fn test_compile_choose_true_predicate() {
    // CHOOSE x \in {1, 2} : TRUE
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Const {
            value: Value::Bool(true),
            ty: TirType::Bool,
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("CHOOSE with TRUE predicate should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseBegin { .. })));
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseNext { .. })));
}

#[test]
fn test_compile_choose_with_and_predicate() {
    // CHOOSE x \in {1, 2, 3} : x > 0 /\ x < 3
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::BoolBinOp {
            left: Box::new(spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirCmpOp::Gt,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(0),
                    ty: TirType::Int,
                })),
            })),
            op: TirBoolOp::And,
            right: Box::new(spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirCmpOp::Lt,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                })),
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("CHOOSE with AND predicate should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseBegin { .. })));
    // AND should use short-circuit JumpFalse
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::JumpFalse { .. })));
}

#[test]
fn test_compile_choose_with_or_predicate() {
    // CHOOSE x \in {1, 2, 3} : x = 1 \/ x = 3
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::BoolBinOp {
            left: Box::new(spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirCmpOp::Eq,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                })),
            })),
            op: TirBoolOp::Or,
            right: Box::new(spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirCmpOp::Eq,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                })),
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("CHOOSE with OR predicate should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseBegin { .. })));
    // OR should use short-circuit JumpTrue
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::JumpTrue { .. })));
}

#[test]
fn test_compile_choose_with_if_then_else_predicate() {
    // CHOOSE x \in {1, 2, 3} : IF x > 1 THEN x < 3 ELSE FALSE
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::If {
            cond: Box::new(spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirCmpOp::Gt,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                })),
            })),
            then_: Box::new(spanned(TirExpr::Cmp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirCmpOp::Lt,
                right: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(3),
                    ty: TirType::Int,
                })),
            })),
            else_: Box::new(spanned(TirExpr::Const {
                value: Value::Bool(false),
                ty: TirType::Bool,
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("CHOOSE with IF-THEN-ELSE predicate should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseBegin { .. })));
    // IF-THEN-ELSE should produce Jump and JumpFalse
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Jump { .. })));
}

#[test]
fn test_compile_nested_choose() {
    // CHOOSE x \in {1, 2} : x = CHOOSE y \in {1, 2} : y > 1
    let mut compiler = BytecodeCompiler::new();
    let inner_choose = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "y".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("y")))),
            op: TirCmpOp::Gt,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(1),
                ty: TirType::Int,
            })),
        })),
    });
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirCmpOp::Eq,
            right: Box::new(inner_choose),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("nested CHOOSE should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Should have 2 ChooseBegin and 2 ChooseNext (nested).
    let choose_begins = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::ChooseBegin { .. }))
        .count();
    let choose_nexts = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::ChooseNext { .. }))
        .count();
    assert_eq!(choose_begins, 2, "nested CHOOSE should have 2 ChooseBegin");
    assert_eq!(choose_nexts, 2, "nested CHOOSE should have 2 ChooseNext");
}

#[test]
fn test_compile_choose_without_domain_returns_error() {
    // CHOOSE x : x > 0  (no domain — unbounded CHOOSE)
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: None,
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirCmpOp::Gt,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(0),
                ty: TirType::Int,
            })),
        })),
    });
    let result = compiler.compile_expression("test", &expr);
    assert!(
        result.is_err(),
        "CHOOSE without domain should return CompileError"
    );
}

#[test]
fn test_compile_choose_with_range_domain() {
    // CHOOSE x \in 1..5 : x > 3
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Choose {
        var: TirBoundVar {
            name: "x".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::Range {
                lo: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                })),
                hi: Box::new(spanned(TirExpr::Const {
                    value: Value::SmallInt(5),
                    ty: TirType::Int,
                })),
            }))),
            pattern: None,
        },
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirCmpOp::Gt,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(3),
                ty: TirType::Int,
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("CHOOSE with range domain should compile");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Should have Range opcode for the domain
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::Range { .. })));
    assert!(func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ChooseBegin { .. })));
}

#[test]
fn test_compile_multi_var_forall() {
    // \A x \in {1, 2}, y \in {3, 4} : x + y > 0
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Forall {
        vars: vec![
            TirBoundVar {
                name: "x".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
            TirBoundVar {
                name: "y".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(3),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(4),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
        ],
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::ArithBinOp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirArithOp::Add,
                right: Box::new(spanned(TirExpr::Name(ident_name("y")))),
            })),
            op: TirCmpOp::Gt,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(0),
                ty: TirType::Int,
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("multi-variable FORALL should compile via nesting");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Should have two nested ForallBegin/ForallNext pairs.
    let forall_begins = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::ForallBegin { .. }))
        .count();
    let forall_nexts = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::ForallNext { .. }))
        .count();
    assert_eq!(forall_begins, 2, "should have 2 nested ForallBegin");
    assert_eq!(forall_nexts, 2, "should have 2 nested ForallNext");
}

#[test]
fn test_compile_multi_var_exists() {
    // \E x \in {1, 2}, y \in {3, 4} : x + y = 5
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::Exists {
        vars: vec![
            TirBoundVar {
                name: "x".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
            TirBoundVar {
                name: "y".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(3),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(4),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
        ],
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::ArithBinOp {
                left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
                op: TirArithOp::Add,
                right: Box::new(spanned(TirExpr::Name(ident_name("y")))),
            })),
            op: TirCmpOp::Eq,
            right: Box::new(spanned(TirExpr::Const {
                value: Value::SmallInt(5),
                ty: TirType::Int,
            })),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("multi-variable EXISTS should compile via nesting");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    let exists_begins = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::ExistsBegin { .. }))
        .count();
    assert_eq!(exists_begins, 2, "should have 2 nested ExistsBegin");
}

#[test]
fn test_compile_multi_var_set_builder_flattens_with_big_union() {
    // {x + y : x \in {1, 2}, y \in {3, 4}}
    // Multi-variable SetBuilder desugars via BigUnion:
    // UNION {{x + y : y \in {3,4}} : x \in {1,2}}
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::SetBuilder {
        body: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirArithOp::Add,
            right: Box::new(spanned(TirExpr::Name(ident_name("y")))),
        })),
        vars: vec![
            TirBoundVar {
                name: "x".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
            TirBoundVar {
                name: "y".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(3),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(4),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
        ],
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("multi-variable SetBuilder should compile via BigUnion flattening");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Should have 2 nested SetBuilderBegin loops and 1 BigUnion.
    let set_builder_begins = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::SetBuilderBegin { .. }))
        .count();
    assert_eq!(
        set_builder_begins, 2,
        "should have 2 nested SetBuilderBegin"
    );
    let big_unions = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::BigUnion { .. }))
        .count();
    assert_eq!(big_unions, 1, "should have 1 BigUnion to flatten");
}

#[test]
fn test_compile_multi_var_func_def_tuple_domain() {
    // [x \in {1, 2}, y \in {3, 4} |-> x + y]
    // Multi-variable FuncDef desugars to tuple-domain:
    // [t \in {1,2} \X {3,4} |-> LET x == t[1], y == t[2] IN x + y]
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::FuncDef {
        vars: vec![
            TirBoundVar {
                name: "x".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
            TirBoundVar {
                name: "y".to_string(),
                name_id: tla_core::NameId(0),
                domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(3),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(4),
                        ty: TirType::Int,
                    }),
                ])))),
                pattern: None,
            },
        ],
        body: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(spanned(TirExpr::Name(ident_name("x")))),
            op: TirArithOp::Add,
            right: Box::new(spanned(TirExpr::Name(ident_name("y")))),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("multi-variable FuncDef should compile via tuple-domain desugaring");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    // Should have 1 Times (cross product) and 1 FuncDefBegin.
    let times_count = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::Times { .. }))
        .count();
    assert_eq!(times_count, 1, "should have 1 Times for cross product");
    let func_def_begins = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::FuncDefBegin { .. }))
        .count();
    assert_eq!(func_def_begins, 1, "should have 1 FuncDefBegin");
    // Should have 2 FuncApply for destructuring t[1] and t[2].
    let func_applies = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::FuncApply { .. }))
        .count();
    assert_eq!(
        func_applies, 2,
        "should have 2 FuncApply for tuple destructuring"
    );
}

#[test]
fn test_compile_set_filter_tuple_pattern_destructures_binding() {
    // { <<a, b>> \in S : a < b }
    // The binder is a TUPLE pattern, so the compiler must destructure the
    // current element into per-component registers (a = elem[1], b = elem[2])
    // and resolve the body references `a` / `b` against those registers.
    // Without consuming the pattern this fails with "unresolved identifier 'b'".
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::SetFilter {
        var: TirBoundVar {
            name: "<<a, b>>".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![spanned(
                TirExpr::Tuple(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ]),
            )])))),
            pattern: Some(TirBoundPattern::Tuple(vec![
                ("a".to_string(), tla_core::NameId(0)),
                ("b".to_string(), tla_core::NameId(0)),
            ])),
        },
        body: Box::new(spanned(TirExpr::Cmp {
            left: Box::new(spanned(TirExpr::Name(ident_name("a")))),
            op: TirCmpOp::Lt,
            right: Box::new(spanned(TirExpr::Name(ident_name("b")))),
        })),
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("tuple-pattern SetFilter should compile by destructuring the binder");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::SetFilterBegin { .. })),
        "should emit SetFilterBegin"
    );
    // Two components => two FuncApply (elem[1], elem[2]) for destructuring.
    let func_applies = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::FuncApply { .. }))
        .count();
    assert_eq!(
        func_applies, 2,
        "should emit one FuncApply per tuple component"
    );
}

#[test]
fn test_compile_set_builder_tuple_pattern_destructures_binding() {
    // { a + b : <<a, b>> \in S }
    // Tuple-pattern set comprehension (SlidingPuzzles-style). The body
    // references the destructured components `a` and `b`.
    let mut compiler = BytecodeCompiler::new();
    let expr = spanned(TirExpr::SetBuilder {
        body: Box::new(spanned(TirExpr::ArithBinOp {
            left: Box::new(spanned(TirExpr::Name(ident_name("a")))),
            op: TirArithOp::Add,
            right: Box::new(spanned(TirExpr::Name(ident_name("b")))),
        })),
        vars: vec![TirBoundVar {
            name: "<<a, b>>".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![spanned(
                TirExpr::Tuple(vec![
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(1),
                        ty: TirType::Int,
                    }),
                    spanned(TirExpr::Const {
                        value: Value::SmallInt(2),
                        ty: TirType::Int,
                    }),
                ]),
            )])))),
            pattern: Some(TirBoundPattern::Tuple(vec![
                ("a".to_string(), tla_core::NameId(0)),
                ("b".to_string(), tla_core::NameId(0)),
            ])),
        }],
    });
    let idx = compiler
        .compile_expression("test", &expr)
        .expect("tuple-pattern SetBuilder should compile by destructuring the binder");
    let chunk = compiler.finish();
    let func = chunk.get_function(idx);
    assert!(
        func.instructions
            .iter()
            .any(|op| matches!(op, Opcode::SetBuilderBegin { .. })),
        "should emit SetBuilderBegin"
    );
    let func_applies = func
        .instructions
        .iter()
        .filter(|op| matches!(op, Opcode::FuncApply { .. }))
        .count();
    assert_eq!(
        func_applies, 2,
        "should emit one FuncApply per tuple component"
    );
}

fn projection_index(expr: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::FuncApply {
        func: Box::new(expr),
        arg: Box::new(spanned(TirExpr::Const {
            value: Value::SmallInt(2),
            ty: TirType::Int,
        })),
    })
}

fn projection_call_named(callee: &str, arg: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::Apply {
        op: Box::new(spanned(TirExpr::Name(ident_name(callee)))),
        args: vec![arg],
    })
}

fn projection_filter_body(outer: Spanned<TirExpr>, call_arg: Spanned<TirExpr>) -> Spanned<TirExpr> {
    projection_filter_body_named("Edges", outer, call_arg)
}

fn projection_filter_body_named(
    callee: &str,
    outer: Spanned<TirExpr>,
    call_arg: Spanned<TirExpr>,
) -> Spanned<TirExpr> {
    spanned(TirExpr::In {
        elem: Box::new(spanned(TirExpr::Tuple(vec![
            outer,
            spanned(TirExpr::Name(ident_name("c"))),
        ]))),
        set: Box::new(projection_call_named(callee, call_arg)),
    })
}

fn projection_filter_expr(body: Spanned<TirExpr>) -> Spanned<TirExpr> {
    spanned(TirExpr::SetFilter {
        var: TirBoundVar {
            name: "c".to_string(),
            name_id: tla_core::NameId(0),
            domain: Some(Box::new(spanned(TirExpr::SetEnum(vec![
                spanned(TirExpr::Const {
                    value: Value::SmallInt(1),
                    ty: TirType::Int,
                }),
                spanned(TirExpr::Const {
                    value: Value::SmallInt(2),
                    ty: TirType::Int,
                }),
            ])))),
            pattern: None,
        },
        body: Box::new(body),
    })
}

fn compile_projection_filter(
    body: Spanned<TirExpr>,
    callee_body: Spanned<TirExpr>,
    enable_tuple2: bool,
    enable_hoist: bool,
) -> Vec<Opcode> {
    compile_projection_filter_with_precedence(
        body,
        callee_body,
        "Edges",
        std::collections::HashMap::new(),
        std::collections::HashSet::new(),
        enable_tuple2,
        enable_hoist,
    )
}

fn compile_projection_filter_with_precedence(
    body: Spanned<TirExpr>,
    callee_body: Spanned<TirExpr>,
    metadata_name: &str,
    replacements: std::collections::HashMap<String, String>,
    force_external: std::collections::HashSet<String>,
    enable_tuple2: bool,
    enable_hoist: bool,
) -> Vec<Opcode> {
    let expr = projection_filter_expr(body);
    let callees = std::collections::HashMap::from([(
        metadata_name.to_string(),
        CalleeInfo {
            params: vec!["p".to_string()],
            body: std::sync::Arc::new(callee_body),
            ast_body: None,
        },
    )]);
    let mut compiler = BytecodeCompiler::new();
    compiler.set_op_replacements(replacements);
    compiler.set_force_external_ops(force_external);
    if enable_tuple2 {
        compiler.enable_tuple2_set_in();
    }
    if enable_hoist {
        compiler.enable_set_filter_projection_hoist();
    }
    let idx = compiler
        .compile_expression_with_callees(
            "Children",
            &["graph".to_string(), "outer".to_string()],
            &expr,
            &callees,
        )
        .expect("projection filter should compile");
    compiler.finish().get_function(idx).instructions.clone()
}

fn exact_projection_filter_body() -> Spanned<TirExpr> {
    projection_filter_body(
        spanned(TirExpr::Name(ident_name("outer"))),
        spanned(TirExpr::Name(ident_name("graph"))),
    )
}

fn exact_projection_callee_body() -> Spanned<TirExpr> {
    projection_index(spanned(TirExpr::Name(ident_name("p"))))
}

#[test]
fn test_set_filter_projection_hoist_emits_one_projection_preheader() {
    let instructions = compile_projection_filter(
        exact_projection_filter_body(),
        exact_projection_callee_body(),
        true,
        true,
    );
    let begin_idx = instructions
        .iter()
        .position(|op| matches!(op, Opcode::SetFilterBegin { .. }))
        .expect("SetFilterBegin");

    let Opcode::LoadImm {
        rd: r_index,
        value: 2,
    } = instructions[begin_idx + 1]
    else {
        panic!("expected projection index preheader: {instructions:?}");
    };
    let Opcode::FuncApply {
        rd: r_set,
        func: 0,
        arg,
    } = instructions[begin_idx + 2]
    else {
        panic!("expected direct projection preheader: {instructions:?}");
    };
    assert_eq!(arg, r_index);

    let Opcode::SetFilterBegin {
        r_binding,
        loop_end,
        ..
    } = instructions[begin_idx]
    else {
        unreachable!();
    };
    let Opcode::Tuple2SetIn {
        first: 1,
        second,
        set,
        ..
    } = instructions[begin_idx + 3]
    else {
        panic!("expected direct tuple membership: {instructions:?}");
    };
    assert_eq!(second, r_binding);
    assert_eq!(set, r_set);

    let next_idx = begin_idx + 4;
    let Opcode::LoopNext { loop_begin, .. } = instructions[next_idx] else {
        panic!("expected LoopNext after membership: {instructions:?}");
    };
    assert_eq!(
        (next_idx as i64) + i64::from(loop_begin),
        (begin_idx + 3) as i64,
        "later iterations must skip the projection preheader"
    );
    assert_eq!(
        (begin_idx as i64) + i64::from(loop_end),
        (next_idx + 1) as i64,
        "an empty domain must skip the projection preheader and membership"
    );
    assert!(
        !instructions
            .iter()
            .any(|op| matches!(op, Opcode::Call { .. })),
        "the exact callee projection must be emitted directly: {instructions:?}"
    );
}

#[test]
fn test_set_filter_projection_hoist_requires_tuple2_fusion() {
    let instructions = compile_projection_filter(
        exact_projection_filter_body(),
        exact_projection_callee_body(),
        false,
        true,
    );
    assert!(instructions
        .iter()
        .any(|op| matches!(op, Opcode::Call { .. })));
    assert!(instructions
        .iter()
        .any(|op| matches!(op, Opcode::TupleNew { count: 2, .. })));
    assert!(!instructions
        .iter()
        .any(|op| matches!(op, Opcode::Tuple2SetIn { .. })));
}

#[test]
fn test_set_filter_projection_hoist_refusals_preserve_bytecode() {
    let non_projection_callee = spanned(TirExpr::Cmp {
        left: Box::new(exact_projection_callee_body()),
        op: TirCmpOp::Eq,
        right: Box::new(exact_projection_callee_body()),
    });
    let computed_arg = projection_filter_body(
        spanned(TirExpr::Name(ident_name("outer"))),
        projection_index(spanned(TirExpr::Name(ident_name("graph")))),
    );
    let computed_outer = projection_filter_body(
        projection_index(spanned(TirExpr::Name(ident_name("graph")))),
        spanned(TirExpr::Name(ident_name("graph"))),
    );
    let binder_dependent_rhs = projection_filter_body(
        spanned(TirExpr::Name(ident_name("outer"))),
        spanned(TirExpr::Name(ident_name("c"))),
    );
    let shadowed_outer = projection_filter_body(
        spanned(TirExpr::Name(ident_name("c"))),
        spanned(TirExpr::Name(ident_name("graph"))),
    );

    for (body, callee_body) in [
        (exact_projection_filter_body(), non_projection_callee),
        (computed_arg, exact_projection_callee_body()),
        (computed_outer, exact_projection_callee_body()),
        (binder_dependent_rhs, exact_projection_callee_body()),
        (shadowed_outer, exact_projection_callee_body()),
    ] {
        let baseline = compile_projection_filter(body.clone(), callee_body.clone(), true, false);
        let refused = compile_projection_filter(body, callee_body, true, true);
        assert_eq!(
            refused, baseline,
            "a refused shape must preserve historical bytecode exactly"
        );
    }
}

#[test]
fn test_set_filter_projection_hoist_refuses_state_and_constant_operands() {
    let state_outer = spanned(TirExpr::Name(TirNameRef {
        name: "state_outer".to_string(),
        name_id: tla_core::NameId(0),
        kind: TirNameKind::StateVar { index: 0 },
        ty: TirType::Dyn,
    }));
    let constant_graph = spanned(TirExpr::Const {
        value: Value::tuple([Value::empty_set(), Value::empty_set()]),
        ty: TirType::Dyn,
    });

    for body in [
        projection_filter_body(state_outer, spanned(TirExpr::Name(ident_name("graph")))),
        projection_filter_body(spanned(TirExpr::Name(ident_name("outer"))), constant_graph),
    ] {
        let baseline =
            compile_projection_filter(body.clone(), exact_projection_callee_body(), true, false);
        let refused = compile_projection_filter(body, exact_projection_callee_body(), true, true);
        assert_eq!(
            refused, baseline,
            "state/constant operands must preserve historical bytecode"
        );
        assert!(refused.iter().any(|op| matches!(op, Opcode::Call { .. })));
    }
}

#[test]
fn test_set_filter_projection_hoist_refuses_parameterized_let_shadow() {
    fn compile(enable_hoist: bool) -> Vec<Opcode> {
        let filter = projection_filter_expr(exact_projection_filter_body());
        let expr = spanned(TirExpr::Let {
            defs: vec![TirLetDef {
                name: "Edges".to_string(),
                name_id: tla_core::NameId(0),
                params: vec!["p".to_string()],
                body: exact_projection_callee_body(),
            }],
            body: Box::new(filter),
        });
        let callees = std::collections::HashMap::from([(
            "Edges".to_string(),
            CalleeInfo {
                params: vec!["p".to_string()],
                body: std::sync::Arc::new(exact_projection_callee_body()),
                ast_body: None,
            },
        )]);
        let mut compiler = BytecodeCompiler::new();
        compiler.enable_tuple2_set_in();
        if enable_hoist {
            compiler.enable_set_filter_projection_hoist();
        }
        let idx = compiler
            .compile_expression_with_callees(
                "Children",
                &["graph".to_string(), "outer".to_string()],
                &expr,
                &callees,
            )
            .expect("LET-shadowed projection filter should compile");
        compiler.finish().get_function(idx).instructions.clone()
    }

    let baseline = compile(false);
    let refused = compile(true);
    assert_eq!(
        refused, baseline,
        "a parameterized LET-local Edges must retain local Call resolution"
    );
    assert!(refused.iter().any(|op| matches!(op, Opcode::Call { .. })));
}

#[test]
fn test_set_filter_projection_hoist_respects_apply_precedence() {
    let outer = spanned(TirExpr::Name(ident_name("outer")));
    let graph = spanned(TirExpr::Name(ident_name("graph")));

    // Fixed builtins are selected before same-name global callees by the
    // normal Apply path. The hoist must not reinterpret that call as metadata.
    let builtin_body = projection_filter_body_named("Len", outer.clone(), graph.clone());
    let builtin_baseline = compile_projection_filter_with_precedence(
        builtin_body.clone(),
        exact_projection_callee_body(),
        "Len",
        std::collections::HashMap::new(),
        std::collections::HashSet::new(),
        true,
        false,
    );
    let builtin_refused = compile_projection_filter_with_precedence(
        builtin_body,
        exact_projection_callee_body(),
        "Len",
        std::collections::HashMap::new(),
        std::collections::HashSet::new(),
        true,
        true,
    );
    assert_eq!(builtin_refused, builtin_baseline);
    assert!(builtin_refused
        .iter()
        .any(|op| matches!(op, Opcode::CallBuiltin { .. })));

    // A force-external name must retain interpreter callback dispatch.
    let forced = std::collections::HashSet::from(["Edges".to_string()]);
    let forced_baseline = compile_projection_filter_with_precedence(
        exact_projection_filter_body(),
        exact_projection_callee_body(),
        "Edges",
        std::collections::HashMap::new(),
        forced.clone(),
        true,
        false,
    );
    let forced_refused = compile_projection_filter_with_precedence(
        exact_projection_filter_body(),
        exact_projection_callee_body(),
        "Edges",
        std::collections::HashMap::new(),
        forced,
        true,
        true,
    );
    assert_eq!(forced_refused, forced_baseline);
    assert!(forced_refused
        .iter()
        .any(|op| matches!(op, Opcode::CallExternal { .. })));

    // Config replacement changes which global definition normal Apply calls;
    // even an equally-shaped target stays on that established Call path.
    let replacements =
        std::collections::HashMap::from([("Edges".to_string(), "CfgEdges".to_string())]);
    let replacement_baseline = compile_projection_filter_with_precedence(
        exact_projection_filter_body(),
        exact_projection_callee_body(),
        "CfgEdges",
        replacements.clone(),
        std::collections::HashSet::new(),
        true,
        false,
    );
    let replacement_refused = compile_projection_filter_with_precedence(
        exact_projection_filter_body(),
        exact_projection_callee_body(),
        "CfgEdges",
        replacements,
        std::collections::HashSet::new(),
        true,
        true,
    );
    assert_eq!(replacement_refused, replacement_baseline);
    assert!(replacement_refused
        .iter()
        .any(|op| matches!(op, Opcode::Call { .. })));

    // The active filter binder also shadows a same-name global callee. Normal
    // Apply treats it as a first-class value, never as global metadata.
    let shadowed_callee_body = projection_filter_body_named("c", outer, graph);
    let shadowed_callee_baseline = compile_projection_filter_with_precedence(
        shadowed_callee_body.clone(),
        exact_projection_callee_body(),
        "c",
        std::collections::HashMap::new(),
        std::collections::HashSet::new(),
        true,
        false,
    );
    let shadowed_callee_refused = compile_projection_filter_with_precedence(
        shadowed_callee_body,
        exact_projection_callee_body(),
        "c",
        std::collections::HashMap::new(),
        std::collections::HashSet::new(),
        true,
        true,
    );
    assert_eq!(shadowed_callee_refused, shadowed_callee_baseline);
    assert!(shadowed_callee_refused
        .iter()
        .any(|op| matches!(op, Opcode::ValueApply { .. })));
}
