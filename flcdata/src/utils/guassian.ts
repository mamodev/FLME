// Function to generate a standard normal random number (mean 0, std dev 1)
export function gaussianRandom() {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while (v === 0) v = Math.random();
    let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return num;
  }
  
  // Function to generate a normal random number with specified mean and standard deviation
export function gaussianRandomWithParams(mean: number, stdDev: number) {
    return mean + gaussianRandom() * stdDev;
}