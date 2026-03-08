import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "halo-forge public",
  description: "Public training, monitoring, results, and readiness surface for halo-forge.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
