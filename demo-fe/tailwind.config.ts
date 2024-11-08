import withMT from "@material-tailwind/react/utils/withMT";
import type { Config } from "tailwindcss";
import { cjsInterop } from "vite-plugin-cjs-interop";

export default withMT({
  content: ["./app/**/{**,.client,.server}/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [
    cjsInterop({
      dependencies: ["@material-tailwind/react"],
    }),
  ],
} satisfies Config);
