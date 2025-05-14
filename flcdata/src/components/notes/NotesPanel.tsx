import { Stack } from "@mui/material";
import Markdown from "react-markdown";
import { apiurl } from "../../api/backend";
import { useSynchedFile } from "../../api/useSynchedFile";


export function NotesPanel() {
  const file = useSynchedFile(".notes/notes.md");

  return (
    <Stack direction={"row"} flex={1} spacing={2} justifyContent={"center"} >

        <textarea
          style={{
            resize: "none",
            flex: 1,
            maxWidth: "900px",
            fontFamily: "monospace",
            fontSize: "16px",
            padding: "10px",
            borderRadius: "4px",
            border: "1px solid #ccc",
          }}
          value={file.content}
          onChange={(e) => {
            file.onChange(e.target.value);
          }}
          placeholder="Write your notes here..."
        />
      
      <div className="markdown-root"  style={{  position: "relative",  flex:1, overflow: "auto",}}>
        <Markdown
          urlTransform={(absoluteUrl) => {
            if (absoluteUrl.startsWith("assets://")) {
              // Convert the assets:// URL to a file:// URL
              const filePath = absoluteUrl.replace("assets://", apiurl("serve-static/.assets/"));
            
              return filePath;
            }

            return absoluteUrl;
          }}
        
        >{file.content}</Markdown>
      </div>
    </Stack>
  );
}
