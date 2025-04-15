type Primitive =
  | null
  | undefined
  | string
  | number
  | boolean
  | symbol
  | bigint;

export const isObjectType = (value: unknown): value is object =>
    typeof value === 'object';

export const isObject = <T extends object>(value: unknown): value is T =>
    !isNullOrUndefined(value) &&
    !Array.isArray(value) &&
    isObjectType(value) &&
    !isDateObject(value);

export const isNullOrUndefined = (value: unknown): value is null | undefined => value == null;

export const isPrimitive = (value: unknown): value is Primitive =>
    isNullOrUndefined(value) || !isObjectType(value);

export const isDateObject = (value: unknown): value is Date => value instanceof Date;

export default function deepEqual(object1: any, object2: any) {
    if (isPrimitive(object1) || isPrimitive(object2)) {
      return object1 === object2;
    }
  
    if (isDateObject(object1) && isDateObject(object2)) {
      return object1.getTime() === object2.getTime();
    }
  
    const keys1 = Object.keys(object1);
    const keys2 = Object.keys(object2);
  
    if (keys1.length !== keys2.length) {
      return false;
    }
  
    for (const key of keys1) {
      const val1 = object1[key];
  
      if (!keys2.includes(key)) {
        return false;
      }
  
      if (key !== 'ref') {
        const val2 = object2[key];
  
        if (
          (isDateObject(val1) && isDateObject(val2)) ||
          (isObject(val1) && isObject(val2)) ||
          (Array.isArray(val1) && Array.isArray(val2))
            ? !deepEqual(val1, val2)
            : val1 !== val2
        ) {
          return false;
        }
      }
    }
  
    return true;
  }