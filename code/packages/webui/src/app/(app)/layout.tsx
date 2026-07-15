"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ReduxProvider } from "@/components/providers/redux-provider";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { SiteHeader } from "@/components/layout/site-header";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { authService } from "@/services/AuthService";
import { useAppDispatch } from "@/store/hooks";
import { setUser } from "@/store/authSlice";

function AuthGuard({ children }: { children: React.ReactNode }) {
  const dispatch = useAppDispatch();
  const router = useRouter();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    authService
      .me()
      .then((user) => {
        dispatch(setUser(user));
        setChecking(false);
      })
      .catch(() => router.push("/login"));
  }, [dispatch, router]);

  if (checking) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <SidebarProvider>
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader />
        <div className="flex flex-1 flex-col">
          <main className="flex flex-1 flex-col gap-4 md:gap-6">{children}</main>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <ReduxProvider>
      <AuthGuard>{children}</AuthGuard>
    </ReduxProvider>
  );
}
