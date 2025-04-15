import {
  Button,
  Dialog,
  IconButton,
  Stack,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import {
  Pipeline,
  useDeletePipeline,
  usePipelineLog,
  usePipelines,
  useRerunPipeline,
} from "../api/pipelines";
import { DataGrid, GridActionsCellItem, GridColDef } from "@mui/x-data-grid";
import { Delete, ReplayOutlined } from "@mui/icons-material";
import React from "react";

function useColumns(
  onDelete: (row: Pipeline) => void,
  onRerun: (row: Pipeline) => void
): GridColDef<Pipeline>[] {
  return [
    { field: "id", headerName: "ID", width: 90 },
    { field: "dex", headerName: "Name", flex: 1 },
    {
      field: "start_time",
      headerName: "Start Time",
      valueGetter: (time: number) => {
        const date = new Date(time * 1000);
        return date.toLocaleString();
      },

      width: 200,
    },
    {
      field: "last_status_change",
      headerName: "Latest Update",
      width: 200,
      valueGetter: (time: number) => {
        const date = new Date(time * 1000);
        return date.toLocaleString();
      },
    },
    { field: "status", headerName: "Stats", width: 150 },
    {
      field: "actions",
      headerName: "Actions",
      width: 150,
      type: "actions",
      getActions: ({ row }) => [
        <GridActionsCellItem
          icon={<Delete />}
          label="Delete"
          onClick={() => {
            onDelete(row);
          }}
        />,
        <GridActionsCellItem
          disabled={row.status === "RUNNING"}
          icon={<ReplayOutlined />}
          label="Reload"
          onClick={() => {
            onRerun(row);
          }}
        />,
      ],
    },
  ];
}

export function PipelinesPanel() {
  const pipelines = usePipelines();

  const deleteFn = useDeletePipeline();
  const rerunFn = useRerunPipeline();
  const [selectedRow, setSelectedRow] = React.useState<Pipeline | null>(null);

  const columns = useColumns(
    (row) => {
      if (confirm(`Are you sure you want to delete pipeline ${row.id}?`)) {
        deleteFn.mutate(row.id);
      }
    },
    (row) => {
      if (confirm(`Are you sure you want to rerun pipeline ${row.id}?`)) {
        rerunFn.mutate(row.id);
      }
    }
  );

  const refetch = pipelines.refetch;
  const running = (pipelines.data || []).some(
    (pipeline) => pipeline.status === "RUNNING"
  );

  React.useEffect(() => {
    if (running) {
      const interval = setInterval(() => {
        refetch();
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [running]);

  if (pipelines.isLoading) {
    return <div>Loading...</div>;
  }

  if (pipelines.isError) {
    return <div>Error: {pipelines.error.message}</div>;
  }

  if (!pipelines.data) {
    return <div>Error: data is undefined</div>;
  }

  const handleDeleteAll = () => {
    if (confirm("Are you sure you want to delete all pipelines?")) {
      pipelines.data.forEach((row) => {
        deleteFn.mutate(row.id);
      });
    }
  }

  const sorted_pipelines = [...pipelines.data].sort((a, b) => {
    return (
      b.last_status_change -
      a.last_status_change
    );
  });

  return (
    <Stack spacing={2} p={1} sx={{ width: "100%" }}>
      <Stack direction="row" spacing={2} justifyContent="space-between">
        <Typography variant="h5">Pipelines</Typography>

        <Stack direction="row" spacing={2}>
         

          <Button
            startIcon={<ReplayOutlined />}
            onClick={() => pipelines.refetch()}
          >
            Reload
          </Button>

          <Button
            variant="outlined"
            color="error"
            startIcon={<Delete />}
            onClick={handleDeleteAll}
            >
                Delete All
            </Button>
        </Stack>
      </Stack>
      <DataGrid
        rows={sorted_pipelines}
        columns={columns}
        onRowClick={({ row }) => {
          setSelectedRow(row);
        }}
      />

      {selectedRow && (
        <Dialog
          open={true}
          onClose={() => setSelectedRow(null)}
          fullWidth
          maxWidth="md"
        >
          <Logs id={selectedRow.id} />
        </Dialog>
      )}
    </Stack>
  );
}

function Logs({ id }: { id: string }) {
  const logs = usePipelineLog(id);

  const [tabIndex, setTabIndex] = React.useState(0);

  return (
    <Stack>
      <Stack direction="row" spacing={2} justifyContent="space-between">
        <Tabs value={tabIndex} onChange={(e, v) => setTabIndex(v)}>
          <Tab label="Stdout" />
          <Tab label="Stderr" />
        </Tabs>

        <IconButton
          onClick={() => {
            logs.refetch();
          }}
        >
          <ReplayOutlined />
        </IconButton>
      </Stack>

      <Stack
        sx={{
          backgroundColor: "#000",
          color: "#fff",
          padding: 2,
          height: "700px",
          overflowY: "scroll",
        }}
      >
        {tabIndex === 0 &&
          logs.data?.stdout.map((line, i) => {
            return (
              <Typography key={i} variant="body1">
                {line}
              </Typography>
            );
          })}

        {tabIndex === 1 &&
          logs.data?.stderr.map((line, i) => {
            return (
              <Typography key={i} variant="body1">
                {line}
              </Typography>
            );
          })}
      </Stack>
    </Stack>
  );
}
