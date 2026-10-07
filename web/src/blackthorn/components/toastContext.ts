import { createContext, useContext } from "react";

export type ToastKind = "info" | "error" | "success";
export type ToastFn = (text: string, kind?: ToastKind) => void;

export const ToastContext = createContext<ToastFn>(() => {});
export const useToast = () => useContext(ToastContext);
