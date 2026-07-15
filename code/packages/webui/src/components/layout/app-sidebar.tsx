"use client";

import * as React from "react";
import Link from "next/link";
import { BarChartIcon, ClipboardListIcon, HelpCircleIcon, LayoutDashboardIcon, SettingsIcon } from "lucide-react";

import { NavMain } from "@/components/layout/nav-main";
import { NavSecondary, NavtItem } from "@/components/layout/nav-secondary";
import { NavUser } from "@/components/layout/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useAppSelector } from "@/store/hooks";

const navMain = [
  { title: "Dashboard", url: "/dashboard", icon: LayoutDashboardIcon },
  { title: "Cases", url: "/cases", icon: ClipboardListIcon },
];

const navSecondary: NavtItem[] = [
  // { title: "Settings", url: "/settings", icon: SettingsIcon },
];

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const user = useAppSelector((s) => s.auth.user);
  const displayName = user?.email?.split("@")[0] ?? "User";

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild className="data-[slot=sidebar-menu-button]:!p-1.5">
              <Link href="/dashboard">
                <span className="text-base font-bold">UrgeNurse</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <NavMain items={navMain} />
        <NavSecondary items={navSecondary} className="mt-auto" />
      </SidebarContent>

      <SidebarFooter>{user && <NavUser user={{ name: displayName, email: user.email }} />}</SidebarFooter>
    </Sidebar>
  );
}
