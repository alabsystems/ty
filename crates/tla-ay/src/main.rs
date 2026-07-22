use ay_dpll::api::{Logic, SolveResult, Solver, Sort};

fn main() {
    let mut solver = Solver::try_new(Logic::QfAuflia).unwrap();

    let int_sort = Sort::Int;

    let false_val = solver.bool_const(false);
    let true_val = solver.bool_const(true);

    // set = store(store(store(const false, 1, true), 2, true), 3, true)
    let empty_set = solver.try_const_array(int_sort.clone(), false_val).unwrap();
    let e1 = solver.int_const(1);
    let e2 = solver.int_const(2);
    let e3 = solver.int_const(3);

    let mut set = empty_set;
    set = solver.try_store(set, e1, true_val).unwrap();
    set = solver.try_store(set, e2, true_val).unwrap();
    set = solver.try_store(set, e3, true_val).unwrap();

    // assert select(set, x)
    let x = solver.declare_const("x", int_sort.clone());
    let member = solver.try_select(set, x).unwrap();
    solver.try_assert_term(member).unwrap();

    // Explicitly assert that 0 is NOT in the set
    let zero = solver.int_const(0);
    let zero_in_set = solver.try_select(set, zero).unwrap();
    let zero_not_in_set = solver.try_not(zero_in_set).unwrap();
    solver.try_assert_term(zero_not_in_set).unwrap();

    let res = solver.check_sat();
    println!("Result: {:?}", res.result());

    if matches!(res.result(), SolveResult::Sat) {
        let verified_model = solver.model().unwrap();
        let model = verified_model.model();
        if let Some(val) = model.get_int("x") {
            println!("x = {}", val);
        } else {
            println!("x is None");
        }
    }
}
