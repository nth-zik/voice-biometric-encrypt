import type { MetaFunction } from "@remix-run/node";

export const meta: MetaFunction = () => {
  return [
    { title: "Voice Hashing Demo App" },
    { name: "description", content: "Welcome to Voice Hashing!" },
  ];
};

export default function Index() {
  return (
    <div>

    </div>
  );
}

