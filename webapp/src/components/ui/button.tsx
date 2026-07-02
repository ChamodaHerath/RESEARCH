import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-300 disabled:pointer-events-none disabled:opacity-50 cursor-pointer select-none",
  {
    variants: {
      variant: {
        primary:
          "bg-gradient-to-r from-[#3B82F6] to-[#8B5CF6] text-white shadow-[0_0_0_1px_rgba(59,130,246,0.4),0_10px_30px_-8px_rgba(59,130,246,0.55)] hover:shadow-[0_0_0_1px_rgba(139,92,246,0.55),0_14px_40px_-6px_rgba(139,92,246,0.6)] hover:-translate-y-0.5",
        outline:
          "border border-white/15 bg-white/[0.03] text-slate-100 backdrop-blur-md hover:border-[#3B82F6]/60 hover:bg-white/[0.06] hover:-translate-y-0.5",
        ghost: "text-slate-300 hover:text-white hover:bg-white/[0.06]",
        danger:
          "bg-gradient-to-r from-red-500 to-rose-600 text-white shadow-[0_10px_30px_-8px_rgba(239,68,68,0.55)] hover:-translate-y-0.5",
      },
      size: {
        default: "h-11 px-6",
        sm: "h-9 px-4 text-xs",
        lg: "h-14 px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
