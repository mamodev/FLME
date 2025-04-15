export const baseUrl = "http://localhost:5000/api/"

export function apiurl(...paths: string[]): string {
    return baseUrl + paths.join("/")
}