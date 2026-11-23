import { useState } from "react";
import Feed from "./pages/Feed";
import Cluster from "./pages/Cluster";
import AB from "./pages/AB";
import Quality from "./pages/Quality";

const TABS = [
  { id: "feed", label: "Feed", zh: "推荐流" },
  { id: "cluster", label: "Cluster", zh: "集群" },
  { id: "ab", label: "A/B", zh: "实验" },
  { id: "quality", label: "Quality", zh: "质量" },
] as const;
type Tab = (typeof TABS)[number]["id"];

function tabFromHash(): Tab {
  const h = window.location.hash.slice(1);
  return (TABS.find((t) => t.id === h)?.id ?? "feed") as Tab;
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const go = (t: Tab) => {
    setTab(t);
    window.history.replaceState(null, "", `#${t}`);
  };

  return (
    <div className="app">
      <header className="masthead">
        <h1>
          <img className="logo" src="/brand/icon-on-dark.svg" alt="" width={40} height={40} />
          <span className="grad">echostream</span> · 流光回响
        </h1>
        <p className="slogan">Every swipe echoes. 每一次划动，都有回响。</p>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "tab on" : "tab"} onClick={() => go(t.id)}>
              {t.label} <span className="zh">{t.zh}</span>
            </button>
          ))}
        </nav>
      </header>

      <main>
        {/* Feed stays mounted so its before/after diff survives tab switches. */}
        <div hidden={tab !== "feed"}>
          <Feed />
        </div>
        {tab === "cluster" && <Cluster />}
        {tab === "ab" && <AB />}
        {tab === "quality" && <Quality />}
      </main>

      <footer className="footer">
        KuaiRec data © Gao et al., CC BY-SA 4.0 · all numbers measured on one laptop (multi-node on localhost)
      </footer>
    </div>
  );
}
