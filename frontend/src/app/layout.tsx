import type { Metadata } from "next";
import { Archivo, Instrument_Serif, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

/* Three voices.
   grotesque — heavy caps, the structural headline voice
   serif     — the accent words set inside those headlines
   mono      — every measured value, family name and axis tick */
const grotesque = Archivo({
  weight: ["400", "600", "700"], subsets: ["latin"], variable: "--font-grotesque",
});
const serif = Instrument_Serif({
  weight: "400", style: ["normal", "italic"], subsets: ["latin"], variable: "--font-serif",
});
const mono = IBM_Plex_Mono({
  weight: ["400", "500"], subsets: ["latin"], variable: "--font-plex",
});

export const metadata: Metadata = {
  title: "Cipher · Indic Voice Forensic Inspector",
  description: "Detecting AI-generated speech over telephony channels",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en"
          className={`${grotesque.variable} ${serif.variable} ${mono.variable} h-full antialiased`}>
      <body suppressHydrationWarning className="min-h-full">{children}</body>
    </html>
  );
}
