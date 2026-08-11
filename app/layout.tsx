import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PELD-RLDH Risk Calculator",
  description: "Local, browser-based inference using the frozen multicenter PELD-RLDH prediction model.",
  openGraph: {
    title: "PELD-RLDH Risk Calculator",
    description: "Transparent frozen-model prediction after PELD. Research use only.",
    type: "website",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <meta property="og:image" content="./og-card.png" />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:image" content="./og-card.png" />
      </head>
      <body>{children}</body>
    </html>
  );
}
