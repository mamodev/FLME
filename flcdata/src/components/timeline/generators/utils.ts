import { pickRandom, randomInt } from "mathjs";

export function pickNRandomIntegers(
  min: number,
  max: number,
  n: number
): number[] {
  if (n > max - min + 1) {
    throw new Error(
      `Cannot pick ${n} unique integers between ${min} and ${max}.  Not enough unique numbers available.`
    );
  }

  const availableNumbers: number[] = [];
  for (let i = min; i <= max; i++) {
    availableNumbers.push(i);
  }

  const pickedNumbers = pickRandom(availableNumbers, n);

  return pickedNumbers as number[];
}

export function hashClientId(clientId: [number, number]): string {
  return `${clientId[0]}-${clientId[1]}`;
}

export function getAllClientIds(
  client_per_partition: number[]
): [number, number][] {
  const ids: [number, number][] = [];
  for (let i = 0; i < client_per_partition.length; i++) {
    for (let j = 0; j < client_per_partition[i]; j++) {
      ids.push([i, j]);
    }
  }

  return ids;
}


export function shuffleArray<T>(array: T[]): T[] {
  const newArray = [...array]; // Create a copy to avoid modifying the original
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = randomInt(i + 1); // Random index from 0 to i
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]]; // Swap elements
  }
  return newArray;
}