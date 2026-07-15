"use client";

import { usePathname } from "next/navigation";
import { Header } from "./header";

function resolveTitle(pathname: string): string {
  if (pathname === "/dashboard") return "Dashboard";
  if (pathname === "/cases") return "Cases";
  if (pathname.startsWith("/cases/")) return "Case Detail";
  return "UrgeNurse";
}

export function PageTitle() {
  const pathname = usePathname();
  return <Header title={resolveTitle(pathname)} />;
}
