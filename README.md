# Arkworks Circuit Development Tutorial

[Arkworks](https://github.com/arkworks-rs) is a new framework based on Rust for creating zkSNARKs with different proof system (e.g., Groth16, Marlin, ...). Libraries in the arkworks ecosystem provide efficient implementations of all components required to implement zkSNARK applications, from generic finite fields to R1CS constraints for common functionalities.


However, it seems difficult to find online in-depth analysis of how Arkworks works and a boilerplate to describe how to setup a project for beginners. Through the analysis of a [sudoku verification system](https://github.com/marcozecchini/arkworks-tutorial), I would like to gain sufficient knowledge to design a reasonable boilerplate for starting a new arkwork project.

In general, I assume that the reader has no many notions on Rust. Indeed, this is a new language and all its new functionalities and expressions might seem difficult to read for someone that is used to Python or Javascript.

## Constraint systems

![cs](https://hackmd.io/_uploads/BkkuWo7L3.png)

Arkworks allows developers to initialize the circuit in a *host* language which is **Rust**. In this library there is one key type (i.e., the enabler of the framework) which is `Constraint System` that is in charge for mantaining state of the R1CS system while you build it up. More into details, it contains the representation of the matrices of R1CS `A`, `B` and `C` and a `variable` space. Then, it provides three main operations: 1) create a variable (`cs.add_var(p,v) -> id`), 2) create linear combinations from these variables (`cs.zero(); lc.add(c, id) -> lc'` corresponding to `lc' = lc + c *id`) and 3) add constraints to the matrices (`cs.constrain(lc_a, lc_b, lc_c)`).

On top of these three simple operations, the library has been developed to *allocate* in the constraint system basic type variables (*boolean*, *uint*, *uint8*, ...) and verify operations on them with *gadgets*. From the gadgets we add the constraints to the proof system. 

## Arkworks setup
To initialize an Arkworks project in Rust do the following step:

1. Initialize a Rust project with `cargo new <name-of-the-project>`
2. Add the following dependencies in the manifest `Cargo.toml`:
```toml!
[dependencies]
ark-ff = { version = "0.4" }
ark-ec = { version = "0.4" }
ark-bls12-381 = { version = "0.4" }
ark-r1cs-std = { version = "0.4" }
ark-snark = { version = "0.4" }
ark-relations = { version = "0.4" }
ark-groth16 = { version = "0.4" }

[dev-dependencies]
itertools = "0.10.1"
```
* The libraries that we have added are:
    * `ark-r1cs-std` offering standard programming data structure
    * `ark-nonnative` offering algebraic objects 
    * `ark-crypto-primitives` offering cryptographic primitives such as hash functions, merkle trees, commitments, signature verifiers...
    * `ark-ff` providing finite field arithmetics
    * `ark-ec` providing elliptic curve arithmetics
    * `ark-relations` providing constraint system APIs. This is the library containing the `ConstraintSytem` object and all the methods for creating new linear combinations, variables, constraints and enforcing equalities. 
    * `ark-snark` provide all the traits (i.e., interfaces) used by different proof systems.
    * `ark-groth16` provide all the functions to interact with the Groth16 proof system. There are several proof systems available in Arkworks. Indeed, on Arkworks, is easy to switch from on proof system to another. 
![libraries](https://hackmd.io/_uploads/rk9QMiXLh.png)


## Tutorial

In this tutorial, we show how to realize a circuit that verifies that the prover knows a solution for a specific sudoku puzzle. 

We split the tutorial into 5 parts:
1) [***Environmental Setup***](#1-Environmental-setup), we analyze how is the project organized and which modules from arkworks libraries are imported to develop the circuit.
2) [***Variable allocations***](#2-Variable-allocations) where we analyze how to allocate in the variable space of `ConstraintSystem` input, witness and constant variables.
3) [***Constraint addition through Gadgets***](#3-Constraint-addition-through-Gadgets) where we show, given a set of allocated variables, how to build the constraints verifying the correctness of the sudoku solution.
4) [***Interaction with the constrain system and testing***](#4-Interaction-with-the-constraint-system-and-testing) where we investigate how to test and verify the correctness of a witness.
5) [***Building a Groth16 SNARK proof***](#5-Building-a-Groth16-SNARK-proof) where we compute a Groth16 SNARK proof for our circuit.

### 1) Environmental setup

The project has three files: `src/main.rs`, `src/cmp.rs` and `src/alloc.rs`. In `src/main.rs`, we have the main program and the auxiliary function to build the circuit. In `src/alloc.rs`, we define the [traits and implementations](#Inherent-implementations-and-Trait-implementations-~OOP-programming)) to allocate the variable. In `src/cmp.rs`, we have the gadget for verifying the correctness of the witness.

In every module of the project, we start importing the necessary libraries from arkworks along with the modules defined in the same directory. For instance, in `src/main.rs`, we import:
```rust
use ark_ff::{PrimeField};
use ark_r1cs_std::{
    prelude::{Boolean, EqGadget, AllocVar},
    uint8::UInt8,
};
use ark_bls12_381::{Fq as F, Bls12_381, Fr};
use ark_groth16::Groth16;
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError, ConstraintSystemRef, ConstraintSystem};
use ark_snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_std::rand::{SeedableRng, RngCore};

use cmp::CmpGadget;

mod cmp;
mod alloc;
```
Using the keyword [`mod`](#Module-system-management-in-Rust) we import in `src/main.rs` all the defined functions and variable definitions in `src/cmp.rs` and `src/alloc.rs`.

### 2) Variable allocations

In the `src/main.rs` file, we define the two structs that represents the puzzle of the sudoku and its solution.
```rust
pub struct Puzzle<const N: usize, ConstraintF: PrimeField>([[UInt8<ConstraintF>; N]; N]);
pub struct Solution<const N: usize, ConstraintF: PrimeField>([[UInt8<ConstraintF>; N]; N]);
```
These must be allocated in the *variable* space of the Constraint System. Therefore, to do that in `src/alloc.rs` we implement for each of them a trait `AllocVar` with a method `new_variable` (see more details about traits and inherance [here](#Inherent-implementations-and-Trait-implementations-~OOP-programming)).

```rust
/// AllocVar allocates a variable in the constraint system
impl<const N: usize, F: PrimeField> AllocVar<[[u8; N]; N], F> for Puzzle<N, F> {
    fn new_variable<T: Borrow<[[u8; N]; N]>>(
        cs: impl Into<Namespace<F>>,
        f: impl FnOnce() -> Result<T, SynthesisError>,
        mode: AllocationMode,
    ) -> Result<Self, SynthesisError> {
        let cs = cs.into();
        let row = [(); N].map(|_| UInt8::constant(0));
        let mut puzzle = Puzzle([(); N].map(|_| row.clone()));
        let value = f().map_or([[0; N]; N], |f| *f.borrow());
        for (i, row) in value.into_iter().enumerate() {
            for (j, cell) in row.into_iter().enumerate() {
                puzzle.0[i][j] = UInt8::new_variable(cs.clone(), || Ok(cell), mode)?;
            }
        }
        Ok(puzzle)
    }
} 
```
> Note that,`[[u8; N];N]` is a short notation to define a matrix of *unsigned integer of 8 bytes* with rows *N* and *N* columns.

The goal of the `AllocVar` implementation for Puzzle and Solution is to properly invoke the `new_variable` function of `UInt8` type which 
is already implemented in the arkworks framework (line 14). Furthermore, the `new_variable` properly 'keeps track' of each `UInt8` position in the new variable `puzzle`. 

`UInt8`, `UInt` and `Boolean` are basic variable types for working with bits in a circuit. They can be imported in `src/alloc.rs` with ```use ark_r1cs_std::uint8::UInt8;```.

In `new_variable()`, `cs.into()` converts the Puzzle/Solution type into a variable type (line 8). The map function in `let row = [(); N].map(|_| UInt8::constant(0));` returns an array of the same size as `[(); N]` (i.e., an empty array of N elements), with a function applied to each element in order: in particular, it takes as input a [closure](#Closures-anonymous-function) returning an `UInt8` with the constant value of 0. We repeat the same operation in the `let mut puzzle = Puzzle([(); N].map(|_| row.clone()));` creating an empty Puzzle matrix.


`f` function, passed as input, contains the puzzle or the solution (passed within `Ok()` functions in `src/main.rs`) of the sudoku that we retrieve with the `*f.borrow()`. Indeed, in `let value = f().map_or([[0; N]; N], |f| *f.borrow());` we create a matrix with the values of the puzzle or the solution matrix that we will use to populate the `puzzle` variable. 
> Note that `|_|` tells the compiler to infer the returned type by itself. `?` propagates the error, if detected, and `.0` means that we are accessing the first element of the tuple `puzzle`, represented as a matrix.

The `new_variable` method is invoked every time we define a public input (`new_input` method), a witness (`new_witness` method) or a constant variable (`new_constant` method). Indeed, taking a look to the `check_helper` function (see below) in `src/main.rs`, we see that it 1) creates the constraint system, 2) it creates the *input*, Puzzle, and 3) it creates the *witness*, Solution. Finally, 4) it invokes the functions creating the constraints. 

```rust
fn check_helper<const N: usize, ConstraintF: PrimeField>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
) {
    let cs = ConstraintSystem::<ConstraintF>::new_ref(); // 1)
    let puzzle_var = Puzzle::new_input(cs.clone(), || Ok(puzzle)).unwrap(); // 2)
    let solution_var = Solution::new_witness(cs.clone(), || Ok(solution)).unwrap(); // 3)
    check_puzzle_matches_solution(&puzzle_var, &solution_var).unwrap(); // 4)
    check_rows(&solution_var).unwrap();
    assert!(cs.is_satisfied().unwrap());
}
```

### 3) Constraint addition through Gadgets

After having declared and initialized the variables in the proper space of the constraint system, we need to define the linear combinations and the constraints. In `check_helper`, we invoke two functions `check_rows` to check that no number is repeated in a row of the sudoku solution and `check_puzzle_matches_solutions` to check that the sudoku provided solution refers to proper puzzle. 

```rust
fn check_helper<const N: usize, ConstraintF: PrimeField>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
) {
    ...
    check_puzzle_matches_solution(&puzzle_var, &solution_var).unwrap();
    check_rows(&solution_var).unwrap();
    ...
}
```

In `check_rows` we iterate over the rows of the matrix to check whether no value in a row is duplicated. This check is formed of two components:
1) `is_neq` verifying that no `cell` and `prior_cell` are equal. This function outputs a `Boolean` type of the arkworks library.
2) `enforce_equal` on the output of 1) (with True as input) to build a contraint in the proof system actually enforcing that `cell` and `prior_cell` are not the same.

```rust
fn check_rows<const N: usize, ConstraintF: PrimeField>(
    solution: &Solution<N, ConstraintF>,
) -> Result<(), SynthesisError> {
    for row in &solution.0 {
        for (j, cell) in row.iter().enumerate() {
            for prior_cell in &row[0..j] {
                cell.is_neq(&prior_cell)?
                    .enforce_equal(&Boolean::TRUE)?;
            }
        }
    }
    Ok(())
}
```

Similarly, in `check_puzzle_matches_solution` we enforce that the solution `s` is in the range:
1) `is_leq` and `is_geq` verifying that each value in the solution matrix is between 1 and `N`. These functions output a `Boolean` type of the arkworks library.
2) `enforce_equal` on the output of 1) (with True as input) to build a contraint in the proof system enforcing that `s` is between 1 and `N`.

```rust
fn check_puzzle_matches_solution<const N: usize, ConstraintF: PrimeField>(
    puzzle: &Puzzle<N, ConstraintF>,
    solution: &Solution<N, ConstraintF>,
) -> Result<(), SynthesisError> {
    for (p_row, s_row) in puzzle.0.iter().zip(&solution.0) {
        for (p, s) in p_row.iter().zip(s_row) {
            // Ensure that the solution `s` is in the range [1, N]
            s.is_leq(&UInt8::constant(N as u8))?
                .and(&s.is_geq(&UInt8::constant(1))?)?
                .enforce_equal(&Boolean::TRUE)?;

            // Ensure that either the puzzle slot is 0, or that
            // the slot matches equivalent slot in the solution
            (p.is_eq(s)?.or(&p.is_eq(&UInt8::constant(0))?)?)
                .enforce_equal(&Boolean::TRUE)?;
        }
    }
    Ok(())
}

```

In `check_puzzle_matches_solution`, also a similar constraint checks is done to check whether that either the puzzle slot is 0, or that the slot matches equivalent slot in the solution.

In general, we can consider:
1) **Component 1)** as the part of the check that evaluates the logic of the circuit (e.g., one numer is lower/greater than another, two numbers are equal) and outputs a `Boolean` type variable that is `True` if the condition is satisfied or `False` if it is not.
2) **Component 2)** as the part of the check that, taking as input a `Boolean` variable from the evaluation of the logic of the circuit, enforces the circuit correctness (or not) in the constraint system.

In the following subsection, we will dig into the details of both components 1) and 2) to deeply understand how constraints are defined and verified in Arkworks.

#### CmpGadget and Gadgets

Gadgets in arkworks are subcircuits corresponding to useful computations that can be used to build up the full circuit.

`CmpGadget` defined in `src/cmp.rs` is a compound trait inheriting the `R1CSVar` trait and the `EqGadget` trait. `R1CSVar` trait  describes some core functionality that is common to high-level variables, such as `Boolean`s, `FieldVar`s, `GroupVar`s, etc. The functionality are related with equality, cloning, constant type verification in the proof system and the returning of the value. `EqGadget` specifies how to generate constraints that check for equality (or not) for two variables of type `Self` (i.e., of the same type).

There are other types of gadgets available in the library: 
* `ToBitsGadget` and `ToBytesGadget` specifying constraints for conversion to a little-endian bit or byte representation of `self`, respectively.
* `CondSelectGadget` generates constraints for selecting between one of two values.

However, we can define our own gadget to build circuits. This is the case of `CmpGadget` specifying how to generate constraints that compare two variables of type `Self` (i.e., of the same type).

```rust
pub trait CmpGadget<ConstraintF: PrimeField>: R1CSVar<ConstraintF> + EqGadget<ConstraintF> {
    #[inline]
    fn is_geq(&self, other: &Self) -> Result<Boolean<ConstraintF>, SynthesisError> {
        // self >= other => self == other || self > other
        //               => !(self < other)
        self.is_lt(other).map(|b| b.not())
    }

    #[inline]
    fn is_leq(&self, other: &Self) -> Result<Boolean<ConstraintF>, SynthesisError> {
        // self <= other => self == other || self < other
        //               => self == other || other > self
        //               => self >= other
        other.is_geq(self)
    }

    #[inline]
    fn is_gt(&self, other: &Self) -> Result<Boolean<ConstraintF>, SynthesisError> {
        // self > other => !(self == other  || self < other)
        //              => !(self <= other)
        self.is_leq(other).map(|b| b.not())
    }

    fn is_lt(&self, other: &Self) -> Result<Boolean<ConstraintF>, SynthesisError>; 
}
```

`CmpGadget` is implemented for the `UInt8`. It essentially implements `is_lt` function to which are led back all the other function of the trait. Note that:
* If the variables are *constant*, we don't add any constraint on their comparison 
* If the variables are *public input or witness*, we add a constraint in the proof system. To compare unsigned integers, we convert them into bits vectors and, then, we perform all the checks to understand which vector represent the lower number.

```rust
impl<ConstraintF: PrimeField> CmpGadget<ConstraintF> for UInt8<ConstraintF> {
    fn is_lt(&self, other: &Self) -> Result<Boolean<ConstraintF>, SynthesisError> {
        // Determine the variable mode.
        if self.is_constant() && other.is_constant() {
            let self_value = self.value().unwrap();
            let other_value = other.value().unwrap();
            let result = Boolean::constant(self_value < other_value);
            Ok(result)
        } else {
            let diff_bits = self.xor(other)?.to_bits_be()?.into_iter();
            let mut result = Boolean::FALSE;
            let mut a_and_b_equal_so_far = Boolean::TRUE;
            let a_bits = self.to_bits_be()?;
            let b_bits = other.to_bits_be()?;
            for ((a_and_b_are_unequal, a), b) in diff_bits.zip(a_bits).zip(b_bits) {
                let a_is_lt_b = a.not().and(&b)?;
                let a_and_b_are_equal = a_and_b_are_unequal.not();
                result = result.or(&a_is_lt_b.and(&a_and_b_equal_so_far)?)?;
                a_and_b_equal_so_far = a_and_b_equal_so_far.and(&a_and_b_are_equal)?;
            }
            Ok(result)
        }
    }
}
```

> Note that, `zip()` returns a new iterator that will iterate over two other iterators, returning a tuple where the first element comes from the first iterator, and the second element comes from the second iterator. For example:
```rust
let a1 = [1, 2, 3];
let a2 = [4, 5, 6];

let mut iter = a1.iter().zip(a2.iter());

assert_eq!(iter.next(), Some((&1, &4)));
assert_eq!(iter.next(), Some((&2, &5)));
assert_eq!(iter.next(), Some((&3, &6)));
assert_eq!(iter.next(), None);
```
> In our case, we will iterate over the vector containing the xor between the vector `a` and `b`, the vector `a` and the vector `b` simultaneously.


The output of the `is_lt` function, invoked somehow also by the other functions of the trait, is a `Boolean` type of the Arkworks library. We will use it to build a constraint in the proof system.

#### Constraint enforcement

The second component of our verification mechanism invokes the `enforce_equal` function. As we said it adds a constraint in the proof system. It is a method on `self.Boolean` taking as input another `Boolean` value (in our case, it is `Boolean::TRUE`). It recursively calls a stack of functions until we reach the `conditional_enforce_equal` of `EqGadget` implementation for `Boolean` that invokes  `cs.enforce_constraint(lc!() + difference, condition.lc(), lc!())?;` adding the constraint. `enforce_constraint` takes as input three linear combinations.

```rust
impl<F: Field> EqGadget<F> for Boolean<F> {
    #[tracing::instrument(target = "r1cs")]
    fn is_eq(&self, other: &Self) -> Result<Boolean<F>, SynthesisError> {
        // self | other | XNOR(self, other) | self == other
        // -----|-------|-------------------|--------------
        //   0  |   0   |         1         |      1
        //   0  |   1   |         0         |      0
        //   1  |   0   |         0         |      0
        //   1  |   1   |         1         |      1
        Ok(self.xor(other)?.not())
    }

    #[tracing::instrument(target = "r1cs")]
    fn conditional_enforce_equal(
        &self,
        other: &Self,
        condition: &Boolean<F>,
    ) -> Result<(), SynthesisError> {
        use Boolean::*;
        let one = Variable::One;
        let difference = match (self, other) {
            // 1 == 1; 0 == 0
            (Constant(true), Constant(true)) | (Constant(false), Constant(false)) => return Ok(()),
            // false != true
            (Constant(_), Constant(_)) => return Err(SynthesisError::AssignmentMissing),
            // 1 - a
            (Constant(true), Is(a)) | (Is(a), Constant(true)) => lc!() + one - a.variable(),
            // a - 0 = a
            (Constant(false), Is(a)) | (Is(a), Constant(false)) => lc!() + a.variable(),
            // 1 - !a = 1 - (1 - a) = a
            (Constant(true), Not(a)) | (Not(a), Constant(true)) => lc!() + a.variable(),
            // !a - 0 = !a = 1 - a
            (Constant(false), Not(a)) | (Not(a), Constant(false)) => lc!() + one - a.variable(),
            // b - a,
            (Is(a), Is(b)) => lc!() + b.variable() - a.variable(),
            // !b - a = (1 - b) - a
            (Is(a), Not(b)) | (Not(b), Is(a)) => lc!() + one - b.variable() - a.variable(),
            // !b - !a = (1 - b) - (1 - a) = a - b,
            (Not(a), Not(b)) => lc!() + a.variable() - b.variable(),
        };

        if condition != &Constant(false) {
            let cs = self.cs().or(other.cs()).or(condition.cs());
            cs.enforce_constraint(lc!() + difference, condition.lc(), lc!())?;
        }
        Ok(())
    }

    #[tracing::instrument(target = "r1cs")]
    fn conditional_enforce_not_equal(
        &self,
        other: &Self,
        should_enforce: &Boolean<F>,
    ) -> Result<(), SynthesisError> {
        // ...
        // Similar to conditional_enforce_equal function
        // ...
    }
}
```

### 4) Interaction with the constraint system and testing

Until now we have allocated the new variables and added the constraints in the proof system. We can now evaluate valid witnesses for the circuit. This task can be perfomed very easily. We initialize the Constraint System running the command ```let cs = ConstraintSystem::<ConstraintF>::new_ref();``` and evaluate its correctness with the command `cs.is_satisfied().unwrap()`.

```rust
fn check_helper<const N: usize, ConstraintF: PrimeField>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
) {
    let cs = ConstraintSystem::<ConstraintF>::new_ref();
    let puzzle_var = Puzzle::new_input(cs.clone(), || Ok(puzzle)).unwrap(); // || Ok(puzzle) is a closure
    let solution_var = Solution::new_witness(cs.clone(), || Ok(solution)).unwrap();
    check_puzzle_matches_solution(&puzzle_var, &solution_var).unwrap();
    check_rows(&solution_var).unwrap();
    assert!(cs.is_satisfied().unwrap());
}

fn main() {
    use ark_bls12_381::Fq as F;
    // Check that it accepts a valid solution.
    let puzzle = [
        [1, 0],
        [0, 2],
    ];
    let solution = [
        [1, 2],
        [1, 2],
    ];
    // Like in C++ we pass variables by reference and not copying them as parameter in the function.
    // In this way, we are more efficient and are side-effecting on them.
    check_helper::<2, F>(&puzzle, &solution);

    println!("\n=======================================\n");
    check_helper::<2, F>(&puzzle, &solution);
    println!("\n=======================================\n");
    test_groth16::<2>(&puzzle, &solution);
}
```


### 5) Building a Groth16 SNARK proof

We only miss to create the proof. We want to create a proof for the [Groth16 system](https://eprint.iacr.org/2016/260.pdf). Arkworks allows to interact with [different proofs system](https://github.com/arkworks-rs#snark-proving-systems) adopting the R1CS arithmatization with different characteristics (universal setup, faster prover, ...). 

In the last command of the `main()`, we invoke the `test_groth16` function. It takes as input the matrices representing as `u8` the puzzle and the solution of the sudoku. Hence, it takes the same input of `check_helper`.

```rust
fn test_groth16<const N: usize>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
)  {
    // This may not be cryptographically safe, use
    // `OsRng` (for example) in production software.
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(ark_std::test_rng().next_u64());

    println!("Creating parameters...");
    
    // We create empty inputs and witnesses
    let empty_value: u8 = 0;
    let row = [(); N].map(|_| empty_value);
    let puzzle_empty = [(); N].map(|_| row.clone()); 
    println!("{:?}", puzzle_empty);

    // Create parameters for our circuit
    let (pk, vk) = {
        let c = SudokuVerifier::<N> {
            puzzle: &puzzle_empty,
            solution: &puzzle_empty
        };
        Groth16::<Bls12_381>::setup(c, &mut rng).unwrap()
    };

    // ... See later ...

}
```

The first relavant operation performed by `test_groth16` is to setting up the circuit. To perform this operation it creates a variable `c` of type `SudokuVerifier` (it is not important that is the witness yet!) and invokes the function `Groth16::<Bls12_381>::setup(c, &mut rng).unwrap()`. `setup` takes as input the variable `c` and a randomness `rng`. 

`SudokuVerifier` is a struct defined in `main.rs` grouping the puzzle and solution matrices. We need to create it make to implement for it the `ConstraintSynthesizer` traits. In particular, we need to implement the method `generate_constraints` on the `SudokuVerifier` that takes as input a reference to constraint system. The behavior of this function is the same of `check_helper`, thus, we refer to section ["*Constraint addition through Gadgets*"](#3-Constraint-addition-through-Gadgets) for more details about it. 

In general, every circuit, on which will be invoked a specific setup (is it needed also for other proof system?), needs to inherit the `ConstraintSynthesizer` and implement the `generate_constraints` function for its specific circuit adding the correct contraint to the system. Indeed, the cascade of functions generated by `setup` eventually invokes `generate_constraints`. When, `setup` completes returns the proving key and the verification key. 

```rust
struct SudokuVerifier<'a, const N: usize>{
    puzzle: &'a [[u8; N]; N], 
    solution: &'a [[u8; N]; N],
}

impl <'a, const N: usize, F: PrimeField> ConstraintSynthesizer<F> for SudokuVerifier<'a, N> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> { 
        let puzzle_var = Puzzle::new_input(ark_relations::ns!(cs, "puzzle"), || Ok(self.puzzle)).unwrap();
        let solution_var = Solution::new_witness(ark_relations::ns!(cs, "solution"), || Ok(self.solution)).unwrap();
        check_puzzle_matches_solution(&puzzle_var, &solution_var).unwrap();
        check_rows(&solution_var).unwrap();
        Ok(())
    }
}
```

The second relevant operation creates the proof for our witness. We declare our witness `c` as a `SudokuVerifier` struct and we pass it, along with the proving key `pk` and some randomness `rng`, to the `Groth16::<Bls12_381>::prove(&pk, c, &mut rng)` function. 


```rust
fn test_groth16<const N: usize>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
)  {
    
    /// ...

    println!("Creating proofs...");
    let c = SudokuVerifier::<N> {
        puzzle: puzzle,
        solution: solution
    };
        
    let proof = Groth16::<Bls12_381>::prove(&pk, c, &mut rng).unwrap();

    // ...


}
```

To verify that `proof` is correct we need to invoke the function `Groth16<Bls12_381>::verify(&vk, &public_input, &proof)`. The other inputs of the function are the verification key `vk` and a reference to an array containing the public inputs. 

Note that, at a very low level, the `public_input` array must be filled with elements belonging to the prime field $F_q$ underlying the BLS12-381 G1 curve. Each element of $F_q$ represents a bit that represents a portion of the encoding of our input data. In our case, since each Puzzle (of dimension 2x2) is represented by 4 `UInt8` variables and every `UInt8` by 8 `Boolean` variables (representing bits), we need to have an array of public input of 32 elements (i.e., elemonts on $F_q$ representing bits).

```rust
fn test_groth16<const N: usize>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
)  {
    
    /// ...


    println!("Verifying the proof...");

    let public_input =  [
        Fr::from(1),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 1
        Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 0
        Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 0
        Fr::from(0),Fr::from(1),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 2
    ];

    assert!(Groth16::<Bls12_381>::verify(&vk,  
        &public_input, &proof,
        // &[] 
    ).unwrap());

    println!("The proof is correct!");

}
```
###  `Snark` trait
<details>
  <summary>See more</summary>

`Groth16` implements the trait `SNARK` of the `ark_snark` crate (i.e., library).  In general, every proof system built on Arkworks implements this trait making for the developer the interaction with the system very simple and the code very modular. Indeed, this is one of the strength of Arkworks.
    
    
```rust
pub trait SNARK<F: PrimeField> {
    /// The information required by the prover to produce a proof for a specific
    /// circuit *C*.
    type ProvingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// The information required by the verifier to check a proof for a specific
    /// circuit *C*.
    type VerifyingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// The proof output by the prover.
    type Proof: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// This contains the verification key, but preprocessed to enable faster
    /// verification.
    type ProcessedVerifyingKey: Clone + CanonicalSerialize + CanonicalDeserialize;

    /// Errors encountered during setup, proving, or verification.
    type Error: 'static + ark_std::error::Error;

    /// Takes in a description of a computation (specified in R1CS constraints),
    /// and samples proving and verification keys for that circuit.
    fn circuit_specific_setup<C: ConstraintSynthesizer<F>, R: RngCore + CryptoRng>(
        circuit: C,
        rng: &mut R,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), Self::Error>;

    /// Generates a proof of satisfaction of the arithmetic circuit C (specified
    /// as R1CS constraints).
    fn prove<C: ConstraintSynthesizer<F>, R: RngCore + CryptoRng>(
        circuit_pk: &Self::ProvingKey,
        circuit: C,
        rng: &mut R,
    ) -> Result<Self::Proof, Self::Error>;

    /// Checks that `proof` is a valid proof of the satisfaction of circuit
    /// encoded in `circuit_vk`, with respect to the public input `public_input`,
    /// specified as R1CS constraints.
    fn verify(
        circuit_vk: &Self::VerifyingKey,
        public_input: &[F],
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        let pvk = Self::process_vk(circuit_vk)?;
        Self::verify_with_processed_vk(&pvk, public_input, proof)
    }

    /// Preprocesses `circuit_vk` to enable faster verification.
    fn process_vk(
        circuit_vk: &Self::VerifyingKey,
    ) -> Result<Self::ProcessedVerifyingKey, Self::Error>;

    /// Checks that `proof` is a valid proof of the satisfaction of circuit
    /// encoded in `circuit_pvk`, with respect to the public input `public_input`,
    /// specified as R1CS constraints.
    fn verify_with_processed_vk(
        circuit_pvk: &Self::ProcessedVerifyingKey,
        public_input: &[F],
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error>;
}
```

`Groth16` implements also the trait `CircuitSpecificSetupSNARK`. This trait is designed for those type of proof systems requiring a circuit specific setup. Those proof systems that has a universal setup implements the `UniversalSetupSNARK` trait. Both these traits inherits the `SNARK` trait. 
    
</details>
    

## APPENDIX - Rust important concept for using ArkWorks

In general, we suggest to have a basic knowledge of Rust and to read, at least, the first chapters (1,2) of [Programming Rust](https://github.com/francoposa/programming-rust/blob/main/Programming%20Rust%202nd%20Edition.pdf). However, in tihs appendix we provide all the basic reference to understand the tutorial analyzed.

### Rust setup

Install Rust in the following way: 
1. Run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. To check that the installation was successfully run `rustc --version` and `cargo --version` and verify that everything went good.

### Rust relevant facilities to use Arkworks

#### Inherence and Trait implementations (~OOP programming)

<details>
  <summary>See more</summary>
    
  <br>

In Rust, we don’t have `class` keyword (like in Java or Python) but we have `struct` and `impl` so we can mix them to do this:
* struct + impl = class
* struct + trait = interfaces

More in the details, the `impl` keyword in Rust is used to implement some functionality on types. This functionality can include both functions and costs. There are two main kinds of implementations in Rust:

1. ***Inherent implementations***. Inherent implementations, as the name implies, are standalone. They are tied to a single concrete `self` type that is specified after the `impl` keyword. These implementations, unlike standard functions, are always in scope. In the following program, we are adding methods to a Person struct with the impl keyword: 
```rust
struct Person {
  name: String,
  age: u32
}

// Implementing functionality on the Person struct with the impl keyword
impl Person{
  // This method is used to introduce a person
  fn introduction(&self){
    println!("Hello! My name is {} and I am {} years old.", self.name, self.age);
  }

  // This method updates the age of the person on their birthday
  fn birthday(&mut self){
    self.age = self.age + 1
  }
}

fn main() {
  // Instantiating a mutable Person object
  let mut person = Person{name: "Hania".to_string(), age: 23};
  
  // person introduces themself before their birthday
  println!("Introduction before birthday:");
  person.introduction();
  
  // person ages one year on their birthday
  person.birthday();
  
  // person introduces themself after their birthday
  println!("\nIntroduction after birthday:");
  person.introduction();
}
```

Note that, `(&self)` indicates that we are executing the method on the struct we are implementing. 

2. ***Trait implementations***. Rust introduces the concept of *traits*. A trait in Rust is a group of methods that are defined for a particular type. Traits are an abstract definition of shared behavior amongst different types. So, in a way, traits are to Rust what interfaces are to Java or abstract classes are to C++. A trait method is able to access other methods within that trait. We can define a trait following this snippet of code: 
```rust
struct Movie {
    title: String,
    director: String,
    release_year: u32, 
    genre: String
}

// Defining a Details trait by defining the functionality it should include
trait Details {
    fn description(&self) -> String;
    fn years_since_release(&self) -> u32;
}

// Implementing the Details trait on Movie struct
impl Details for Movie{

  // Method returns an overview of the movie
  fn description(&self) -> String{
    return format!("{}, released in {}, is a {} movie directed by {}.", self.title, self.release_year, self.genre, self.director);
  }

  // Method returns the number of years between the writing year of this shot i.e.
  // 2020 and the release year of the movie
  fn years_since_release(&self) -> u32{
    return 2020 - self.release_year;
  }
}

fn main() {
  let movie1 = Movie{
      title: "Titanic".to_string(),
      director: "James Cameron".to_string(),
      release_year: 1997,
      genre: "historical".to_string()
  };
  println!("{}", movie1.description());
  println!("The movie was released {} years ago.", movie1.years_since_release());

  let movie2 = Movie{
      title: "The Dark Knight".to_string(),
      director: "Christopher Nolan".to_string(),
      release_year: 2008,
      genre: "action".to_string()
  };
  println!("\n{}", movie2.description());
  println!("The movie was released {} years ago.", movie2.years_since_release());
}
```
However, we can use already built-in traits like in the following scenario:
```rust
struct Dog {
  name: String,
  age: u32, 
  owner: String
}


// Implementing an in-built trait ToString on the Dog struct
impl ToString for Dog {
  fn to_string(&self) -> String{
    return format!("{} is a {} year old dog who belongs to {}.", self.name, self.age, self.owner);
  }
}

fn main() {
  let dog = Dog{name: "Frodo".to_string(), age: 3, owner: "Maryam".to_string()};
  println!("{}", dog.to_string());
}
```
</details>

#### Module system management in Rust

<details>
  <summary>See more</summary>
    
  <br>
    
At this [link](https://www.sheshbabu.com/posts/rust-module-system/) there is a valid explantion of how modules work in Rust.
Indeed, there is a difference between what we see (file system tree) and what the compiler sees (module tree). In Rust, the compiler only sees the `crate` module which is our `main.rs` file of the project. We need to explicitly build the module tree in Rust - there’s no implicit mapping between file system tree to module tree. To add a file to the module tree, we need to declare that file as a submodule using the `mod` keyword.
    
</details>

#### Closures (~anonymous function)

<details>
  <summary>See more</summary>
    
  <br>

[READ MORE FROM HERE](https://www.cs.brandeis.edu/~cs146a/rust/doc-02-21-2015/book/closures.html)

Rust also allows us to create anonymous functions. Rust's anonymous functions are called closures. By themselves, closures aren't all that interesting, but when you combine them with functions that take closures as arguments, really powerful things are possible.

Let's make a closure:

```rust
let add_one = |x| { 1 + x };

println!("The sum of 5 plus 1 is {}.", add_one(5));
```
We create a closure using the |...| { ... } syntax, and then we create a binding so we can use it later. Note that we call the function using the binding name and two parentheses, just like we would for a named function.

Let's compare syntax. The two are pretty close:

```rust
let add_one = |x: i32| -> i32 { 1 + x };
fn  add_one   (x: i32) -> i32 { 1 + x }
```

</details>
    
#### Generics

<details>
  <summary>See more</summary>
    
  <br>

[READ MORE FROM HERE](https://blog.logrocket.com/understanding-rust-generics/)

Generics are a way to reduce the need to write repetitive code and instead delegate this task to the compiler while also making the code more flexible. Many languages support some way to do this, even though they might call it something different. Using generics, we can write code that can be used with multiple data types without having to rewrite the same code for each data type, making life easier and coding less error-prone.

</details> 
