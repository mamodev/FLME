import { useQuery } from "@tanstack/react-query"

// get current domain / ip address (Exclude port) and set as base url
const domain = window.location.hostname.split(":")[0]

// export const baseUrl = "http://localhost:5555/api/"
export const baseUrl = `http://${domain}:5555/api/`

export function apiurl(...paths: string[]): string {
    return baseUrl + paths.join("/")
}

export async function execPython(code: string, globals: Record<string, any> = {}): Promise<any> {
    return await  fetch(apiurl("exec-python"), {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            code,
            globals,
        }),
    }).then((response) => {
        if (!response.ok) {
            throw new Error("Network response was not ok")
        }
        return response.json()
    })
}

export async function ls(path: string): Promise<string[]> {
    const dir = await execPython(
`
import os

#list all files in a directory and return as list of strings
def ls(path):
    if not os.path.exists(path):
        return []

    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        return [os.path.join(path, f) for f in os.listdir(path)]

    return []

response = ls(path)
`
, {
        path,
    })

    return dir.result as string[]
}

export function useLS(path: string) {
    const res = useQuery({
        queryKey: ["ls", path],
        queryFn: () => ls(path),
    }) 

    return {
        ...res,
        data: res.data ?? [],
    }
}


export async function read(path: string): Promise<string> {
    const file = await execPython(
    `
import os
# read a file and return as string
def read_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return f.read()
    if os.path.isdir(path):
        raise IsADirectoryError(f"Path {path} is a directory")
    
    raise FileNotFoundError(f"Path {path} not found")

response = read_file(path)
    `
    , {
        path,
    })

    return file.result as string
}


export async function write(path: string, content: string): Promise<void> {
    await execPython(
        `
import os
# write a file
def write_file(path, content):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)
response = write_file(path, content)
        `,
        {
            path,
            content,
        }
    ).then(() => {
        console.log("File written successfully")
    }
    ).catch((err) => {
        console.error("Error writing file", err)
    }
    )
}


export async function writeJson(path: string, content: any): Promise<void> {
    return await write(path, JSON.stringify(content))
}

export async function writeBase64(path: string, content: string): Promise<void> {
    return await execPython(
        `
import os
import base64

def write_base64_file(path, content):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(base64.b64decode(content))

response = write_base64_file(path, content)
        `,
        {
            path,
            content,
        }
    )
    
}


export async function updateAt(path: string, content: string, offset: number): Promise<void> {
    await execPython(
        `
import os
# update a file at a given offset
def update_file(path, content, offset):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    if os.path.isfile(path):
        with open(path, "r+") as f:
            f.seek(offset)
            f.write(content)
    if os.path.isdir(path):
        raise IsADirectoryError(f"Path {path} is a directory")
    raise FileNotFoundError(f"Path {path} not found")
response = update_file(path, content, offset)
        `,
        {
            path,
            content,
            offset,
        }
    )
}

