import { ReduxProvider } from "@/components/providers/redux-provider";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <ReduxProvider>
      <div className="min-h-screen flex items-center justify-center bg-gray-50">{children}</div>
    </ReduxProvider>
  );
}
