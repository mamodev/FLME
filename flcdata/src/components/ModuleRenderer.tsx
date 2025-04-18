import { Autocomplete, Checkbox, Stack, TextField, Typography } from "@mui/material";
import {
  FloatParameter,
  IntParameter,
  Module,
  Parameters,
} from "../backend/interfaces";
import { evaluate } from "mathjs";
import React from "react";


type IntParameterProps = {
  label: string;
  parameter: IntParameter;
  value: number;
  onChange: (value: number) => void;
};

function IntParameterRenderer(props: IntParameterProps) {
  const { value, onChange, label } = props;
  const [expression, setExpression] = React.useState(String(value));

  const handleExpressionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newExpression = e.target.value;
    setExpression(newExpression);

    try {
      const newValue = evaluate(newExpression);
      if (typeof newValue === "number" && !isNaN(newValue)) {
        const integerValue = Math.floor(newValue); // Crucial: Convert to integer
        onChange(integerValue);
      }
    } catch (error) {
      // Optionally handle the error, e.g., show an error message to the user
      console.error("Invalid expression:", error);
    }
  };

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Typography variant="body1" textAlign="right" minWidth={150}>
        {label}
      </Typography>

      <TextField
        size="small"
        type="text" // Changed to "text" to allow expressions
        value={expression}
        onChange={handleExpressionChange}
        onBlur={() => {
          // if the expression is invalid, revert to the last valid value
          try {
            const newValue = evaluate(expression);
            if (typeof newValue === "number" && !isNaN(newValue)) {
              const integerValue = Math.floor(newValue); // Crucial: Convert to integer
              onChange(integerValue);
              setExpression(String(integerValue));
            } else {
              setExpression(String(value));
            }
          } catch (error) {
            setExpression(String(value));
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
  const [expression, setExpression] = React.useState(String(value));

  const handleExpressionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newExpression = e.target.value;
    setExpression(newExpression);

    try {
      const newValue = evaluate(newExpression);
      if (typeof newValue === "number" && !isNaN(newValue)) {
        onChange(newValue);
      }
    } catch (error) {
      // Optionally handle the error, e.g., show an error message to the user
      console.error("Invalid expression:", error);
    }
  };

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Typography variant="body1" textAlign="right" minWidth={150}>
        {label}
      </Typography>

      <TextField
        size="small"
        type="text" // Changed to "text" to allow expressions
        value={expression}
        onChange={handleExpressionChange}
        onBlur={() => {
          // if the expression is invalid, revert to the last valid value
          try {
            const newValue = evaluate(expression);
            if (typeof newValue === "number" && !isNaN(newValue)) {
              onChange(newValue);
              setExpression(String(newValue));
            } else {
              setExpression(String(value));
            }
          } catch (error) {
            setExpression(String(value));
          }
        }}
      />
    </Stack>
  );
}

type BoolParameterProps = {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
};


function BoolParameterRenderer(props: BoolParameterProps) {
  const { value, onChange, label } = props;

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Typography variant="body1" textAlign="right" minWidth={150}>
        {label}
      </Typography>

      <Checkbox 
      sx={{padding: 0}}
        size="small"
        checked={value}
        onChange={(e) => {
          onChange(e.target.checked);
        }}
      />
      </Stack>
  );
}


type EnumParameterProps = {
  label: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
};

function EnumParameterRenderer(props: EnumParameterProps) {
  const { value, onChange, label, options } = props;

  return (
    <Stack direction="row" spacing={2} alignItems="center">
      <Typography variant="body1" textAlign="right" minWidth={150}>
        {label}
      </Typography>

      <Autocomplete
        size="small"
        options={options}
        value={value}
        onChange={(event, newValue) => {
          if (newValue) {
            onChange(newValue);
          }
        }}
        renderInput={(params) => (
          <TextField {...params} label={label} variant="outlined" />
        )}
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

  if (parameter.type === "boolean") {
    return (
      <BoolParameterRenderer
        value={value}
        onChange={onChange}
        label={label}
      />
    );
  }

  if (parameter.type === "enum") {
    return (
      <EnumParameterRenderer
        value={value}
        onChange={onChange}
        label={label}
        options={parameter.options}
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
