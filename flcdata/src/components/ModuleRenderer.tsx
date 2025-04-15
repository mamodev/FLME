import { Stack, TextField, Typography } from "@mui/material";
import {
  FloatParameter,
  IntParameter,
  Module,
  Parameters,
} from "../backend/interfaces";

type IntParameterProps = {
  label: string;
  parameter: IntParameter;
  value: number;
  onChange: (value: number) => void;
};

function IntParameterRenderer(props: IntParameterProps) {
  const { value, onChange, label } = props;

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Typography variant="body1" textAlign="right" minWidth={150}>
        {label}
      </Typography>

      <TextField
        size="small"
        type="number"
        value={value}
        onChange={(e) => {
          const newValue = parseInt(e.target.value);
          if (!isNaN(newValue)) {
            onChange(newValue);
          }
        }}
      />
    </Stack>
  );
}

type FloatParameterProps = {
  label: string;
  parameter: FloatParameter;
  value: number;
  onChange: (value: number) => void;
};

function FloatParameterRenderer(props: FloatParameterProps) {
  const { value, onChange, label } = props;

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Typography variant="body1" textAlign="right" minWidth={150}>
        {label}
      </Typography>

      <TextField
        size="small"
        type="number"
        value={value}
        onChange={(e) => {
          const newValue = parseFloat(e.target.value);
          if (!isNaN(newValue)) {
            onChange(newValue);
          }
        }}
      />
    </Stack>
  );
}

type ModuleParameterProps = {
  label: string;
  parameter: Parameters[string];
  value: any;
  onChange: (value: any) => void;
};

function ModuleParameterRenderer(props: ModuleParameterProps) {
  const { parameter, value, onChange, label } = props;

  if (parameter.type === "int") {
    return (
      <IntParameterRenderer
        parameter={parameter}
        value={value}
        onChange={onChange}
        label={label}
      />
    );
  }

  if (parameter.type === "float") {
    return (
      <FloatParameterRenderer
        parameter={parameter}
        value={value}
        onChange={onChange}
        label={label}
      />
    );
  }

  return <p>Not implemented</p>;
}

type ModuleRendererProps = {
  module: Module;
  parameters: Parameters;
  onChange: React.Dispatch<React.SetStateAction<Parameters>>;
};

export function ModuleRenderer(props: ModuleRendererProps) {
  const { module, parameters, onChange } = props;

  return (
    <Stack spacing={.5} 
      sx={{
        "& .MuiInputBase-input": {
          padding: "0px 8px",
        },
      }}
      >

      {/* <Typography variant="h6">{module.name}</Typography>
      <Typography variant="body1">{module.description}</Typography> */}

      {/* <Stack spacing={2}> */}
      {Object.entries(module.parameters).map(([key, parameter]) => {
        return (
          <ModuleParameterRenderer
            key={key}
            label={key
              .replaceAll("_", " ")
              .replaceAll(/([a-z])([A-Z])/g, "$1 $2")}
            parameter={parameter}
            value={parameters[key]}
            onChange={(value) => {
              onChange({
                ...parameters,
                [key]: value,
              });
            }}
          />
        );
      })}
      {/* </Stack>   */}
    </Stack>
  );
}
