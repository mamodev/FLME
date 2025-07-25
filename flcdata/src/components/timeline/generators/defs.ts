import { IModule } from "../types";
import { TLRandomAsync } from "./async";
import { TLSync } from "./sync";
export const TimelineGenerators: IModule[] = [
    TLSync,
    TLRandomAsync,
]