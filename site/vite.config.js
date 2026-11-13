import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// base: './' 让打包产物可以直接用 file:// 或任意子路径打开
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
});
