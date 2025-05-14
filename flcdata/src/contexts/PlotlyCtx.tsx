import React from "react";
import Plotly from "plotly.js";
import { write, writeBase64 } from "../api/backend";
import { ensureExtname, pj } from "../utils/path";

type PlotlyContext = {
    Plotly: typeof Plotly;
}


export type PlotlyData = Plotly.Data 
export type PlotlyLayout = Plotly.Layout
export type PlotlyConfig = Plotly.Config



export const PlotlyCustomDowload: Plotly.ModeBarButton = {
    name: "save-to-asset",
    title: "Save to Asset",
    icon: Plotly.Icons.disk,
    click: async function (gd: Plotly.PlotlyHTMLElement) {
        const base64 = await Plotly.toImage(gd, {
            format: "png",
            width: 800,
            height: 600,
            scale: 2,
        })

        const prefix = "data:image/png;base64,";

        if (!base64.startsWith(prefix)) {
            alert("Failed to get base64 image, base64 prefix not found");
            return;
        }

        const base64Data = base64.slice(prefix.length);
        if (base64Data.length % 4 !== 0) {
            alert("Failed to get base64 image, base64 data is not valid");
            return;
        }

        let name = prompt("Enter a name for the asset", "asset.png");
        if (!name) {
            alert("No name provided");
            return;
        }

    

        name = ensureExtname(name, ".png");


        await writeBase64(pj(".assets", name!), base64Data);
    }
}


const PlotlyCtx = React.createContext<PlotlyContext | null>(null);


export function PlotlyProvider({ children }: { children: React.ReactNode }) {
    return (
        <PlotlyCtx.Provider value={{ Plotly }}>
            {children}
        </PlotlyCtx.Provider>
    );
}

export function usePlotly() {
    return React.useContext(PlotlyCtx)!;
}    

