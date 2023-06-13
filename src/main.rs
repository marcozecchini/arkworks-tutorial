use std::{usize};

use ark_ff::{PrimeField};
use ark_r1cs_std::{
    prelude::{Boolean, EqGadget, AllocVar},
    uint8::UInt8,
};
use ark_bls12_381::{Bls12_381, Fr};


use ark_groth16::Groth16;
use ark_relations::r1cs::{ConstraintSynthesizer, SynthesisError, ConstraintSystemRef};
use ark_snark::{CircuitSpecificSetupSNARK, SNARK};

use ark_std::rand::{SeedableRng, RngCore};
use cmp::CmpGadget;

mod cmp;
mod alloc;

pub struct Puzzle<const N: usize, ConstraintF: PrimeField>([[UInt8<ConstraintF>; N]; N]);
pub struct Solution<const N: usize, ConstraintF: PrimeField>([[UInt8<ConstraintF>; N]; N]);

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

fn test_groth16<const N: usize>(
    puzzle: &[[u8; N]; N],
    solution: &[[u8; N]; N],
)  {
    // This may not be cryptographically safe, use
    // `OsRng` (for example) in production software.
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(ark_std::test_rng().next_u64());

    println!("Creating parameters...");
    
    // We create empty inputs and witnesses
    // e.g., https://github.com/arkworks-rs/groth16/blob/master/tests/mimc.rs
    // https://github.com/arkworks-rs/r1cs-tutorial/blob/5d3a9022fb6deade245505748fd661278e9c0ff9/rollup/src/rollup.rs
    // https://github.com/dev0x1/arkwork-examples
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

    println!("Creating proofs...");
    let c = SudokuVerifier::<N> {
        puzzle: puzzle,
        solution: solution
    };

    println!("Verifying the proof...");
    // Prepare the verification key (for proof verification)
    let pvk = Groth16::<Bls12_381>::process_vk(&vk).unwrap();

    let proof = Groth16::<Bls12_381>::prove(&pk, c, &mut rng).unwrap();
    

    let public_input =  [
        Fr::from(1),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 1
        Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 0
        Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 0
        Fr::from(0),Fr::from(1),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0),Fr::from(0), // 2
    ];

    assert!(Groth16::<Bls12_381>::verify_proof(&pvk, &proof, 
        &public_input 
        // &[] 
    ).unwrap());

    println!("The proof is correct!");

}


fn main()  {
    // Check that it accepts a valid solution.
    let puzzle = [
        [1, 0],
        [0, 2],
    ];
    let solution = [
        [1, 2],
        [1, 2],
    ];

    test_groth16::<2>(&puzzle, &solution);
}
