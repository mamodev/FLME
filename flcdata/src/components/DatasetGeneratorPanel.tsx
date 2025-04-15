import { useGeneratorContext } from "../contexts/GeneratorContext";
import Sidebar from "./Sidebar";
import SidebarToggle from "./SidebarToggle";
import { VisualizationPanel } from "./VisualizationPanel";

export function DatasetGeneratorPanel() {
    
    const { generated } = useGeneratorContext();
    return <>
    
                <Sidebar />
            <SidebarToggle />
            <VisualizationPanel generated={generated} />
    </>
}