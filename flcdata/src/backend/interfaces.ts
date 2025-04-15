type IntParameter = {
    type: "int";
    min: number;
    max: number;
    default: number;
}

type FloatParameter = {
    type: "float";
    min: number;
    max: number;
    default: number;
}

type StringParameter = {
    type: "string";
    default: string;
}

type BooleanParameter = {
    type: "boolean";
    default: boolean;
}

type EnumParameter = {
    type: "enum";
    options: string[];
    default: string;
}

type Parameter = IntParameter | FloatParameter | StringParameter | BooleanParameter | EnumParameter;

type Parameters = {
    [key: string]: Parameter;
}

type Module = {
    name: string;
    description: string;
    parameters: Parameters;
}

type Config = {
    data_generators: Module[];
    distributions: Module[];
    partitioners: Module[];
    plots: Module[];
}

export type { IntParameter, FloatParameter, StringParameter, BooleanParameter, EnumParameter, Parameter, Parameters, Module, Config }