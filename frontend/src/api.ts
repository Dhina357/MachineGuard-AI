export type Machine={machine_id:string;latest_temperature:number;latest_vibration:number;latest_pressure:number;latest_rotational_speed:number;operating_hours:number;latest_risk:number|null;latest_risk_level:string;updated_at:string};
export type Prediction={id:number;reading_id:number;machine_id:string;risk_score:number;risk_level:string;predicted_failure:boolean;model_source:string;created_at:string};
const API=import.meta.env.VITE_API_BASE_URL??"http://localhost:8000";
async function request<T>(path:string):Promise<T>{const response=await fetch(`${API}${path}`);if(!response.ok)throw new Error(`API request failed: ${response.status}`);return response.json() as Promise<T>}
export const getMachines=()=>request<Machine[]>("/api/v1/machines");
export const getPredictions=()=>request<Prediction[]>("/api/v1/predictions?limit=8");
