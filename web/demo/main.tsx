import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import DemoApp from "../src/demo/DemoApp";
import "../src/demo/demo.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <DemoApp />
  </StrictMode>,
);
