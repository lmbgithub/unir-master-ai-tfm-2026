import { Input } from "@/components/ui/input";

interface HeaderProps {
  title: string;
}

export function Header({ title }: HeaderProps) {
  return (
    <header className="h-16 flex items-center justify-between px-6 bg-white border-b border-border shrink-0">
      <h1 className="text-lg font-semibold text-foreground">{title}</h1>
      <Input className="w-72" placeholder="Buscar..." />
    </header>
  );
}
