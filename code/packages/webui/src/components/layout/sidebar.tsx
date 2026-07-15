"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { clearUser } from "@/store/authSlice";
import { authService } from "@/services/AuthService";
import { cn } from "@/lib/utils";

const navLinks = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/cases", label: "Cases" },
];

export function Sidebar() {
  const pathname = usePathname();
  const dispatch = useAppDispatch();
  const router = useRouter();
  const email = useAppSelector((s) => s.auth.user?.email);

  async function handleLogout() {
    await authService.logout().catch(() => undefined);
    dispatch(clearUser());
    router.push("/login");
  }

  return (
    <div className="w-64 h-screen flex flex-col bg-white border-r border-border shrink-0">
      <div className="p-6 border-b border-border">
        <span className="text-xl font-bold text-foreground">UrgeNurse</span>
      </div>

      <nav className="flex-1 p-4 space-y-1">
        {navLinks.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors",
              pathname === href || pathname.startsWith(href + "/")
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            {label}
          </Link>
        ))}
      </nav>

      <div className="p-4 border-t border-border space-y-2">
        <Link
          href="/settings"
          className="flex items-center px-3 py-2 rounded-md text-sm font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
        >
          Settings
        </Link>
        {email && <p className="px-3 py-1 text-xs text-muted-foreground truncate">{email}</p>}
        <button
          onClick={handleLogout}
          className="w-full flex items-center px-3 py-2 rounded-md text-sm font-medium text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors text-left"
        >
          Logout
        </button>
      </div>
    </div>
  );
}
