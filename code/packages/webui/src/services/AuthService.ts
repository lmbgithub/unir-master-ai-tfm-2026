import { apiFetch } from "@/lib/api";
import type { UserResponse } from "@/types/api";

class AuthService {
  async login(email: string, password: string): Promise<UserResponse> {
    return apiFetch<UserResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  async logout(): Promise<void> {
    await apiFetch("/auth/logout", { method: "POST" });
  }

  async me(): Promise<UserResponse> {
    return apiFetch<UserResponse>("/auth/me");
  }
}

export const authService = new AuthService();
