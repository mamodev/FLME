export function basename(path: string): string {
    return path.split("/").pop() ?? path;
}

export function filename(path: string): string {
    return path.split("/").pop() ?? path;
}

export function dirname(path: string): string {
    return path.split("/").slice(0, -1).join("/");
}


export function extname(path: string): string {
    return path.split(".").pop() ?? path;
}

export function ensureExtname(path: string, ext: string): string {
    if (path.endsWith(ext)) {
        return path;
    }

    return path + ext;
}


export function pj(...paths: string[]): string {
    return paths.join("/");
}
