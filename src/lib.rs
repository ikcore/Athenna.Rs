pub mod nn;
pub mod activations;

#[cfg(test)]
mod tests {

	use crate::activations::{sigmoid,tanh};

	#[test]
	fn test_sigmoid() {
		let val = sigmoid(0.5);
		println!("{}", val);
		assert!(val > 0.622 && val < 0.624);
	}

	#[test]
	fn test_tanh() {
		let val = tanh(1.5);
		println!("{}", val);
		assert!(val > 0.904 && val < 0.906);
	}
}
 