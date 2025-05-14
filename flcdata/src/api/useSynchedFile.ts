import React from "react"
import { read, write } from "./backend"
import { debounce } from "@mui/material"

interface UseSynchedFileResult {
    content: string
    onChange: (newContent: string) => Promise<void>
    isLoading: boolean
    error: any
}

export function useSynchedFile(path: string): UseSynchedFileResult {
    const [content, setContent] = React.useState<string>("")
    const [isLoading, setIsLoading] = React.useState<boolean>(true)
    const [error, setError] = React.useState<any>(null)

    const loadFile = React.useCallback(async () => {
        setIsLoading(true)
        setError(null)
        try {
            const fileContent = await read(path)
            setContent(fileContent)
        } catch (e) {
            setError(e)
        } finally {
            setIsLoading(false)
        }
    }, [path])

    React.useEffect(() => {
        void loadFile()
    }, [loadFile])


    const sync = React.useCallback(
        debounce(
            async (newContent: string) => {
                try {
                    await write(path, newContent)
                } catch (e) {
                    console.error("Failed to write to file:", e)
                    setError(e)
                    await loadFile() // Revert to the last known good state
                }
            },
            1000, // Debounce time in milliseconds
        ),
        [path, loadFile]
    )

    const onChange = React.useCallback(
        async (newContent: string) => {
            setContent(newContent) // Optimistically update the state
            sync(newContent) // Trigger the debounced write
           
        },
        [path, loadFile, sync]
    )

    return {
        content,
        onChange,
        isLoading,
        error,
    }
}