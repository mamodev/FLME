import { Module } from "../backend/interfaces";

export function generateDefaultParameters(module: Module): Record<string, any> {
  const parameters: Record<string, any> = {};

  for (const [key, parameter] of Object.entries(module.parameters)) {
    if ('default' in parameter) {
      parameters[key] = parameter.default;
    } else {
      parameters[key] = null;
    }
  }

  return parameters;
}
