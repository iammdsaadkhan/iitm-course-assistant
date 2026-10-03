import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PageWise — Ask your PDFs",
  description: "Upload a PDF and ask questions. Get useful answers with page-level sources.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
