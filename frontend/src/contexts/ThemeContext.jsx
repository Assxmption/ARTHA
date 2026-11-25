import { createContext, useContext, useEffect, useState } from 'react';
import { setPreference, usePreferences } from '../lib/preferences';

const ThemeContext = createContext();

function systemPrefersDark() {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  } catch {
    return true;
  }
}

export function ThemeProvider({ children }) {
  const { theme } = usePreferences(); // 'dark' | 'light' | 'system'
  const [systemDark, setSystemDark] = useState(systemPrefersDark);

  useEffect(() => {
    let mq;
    try { mq = window.matchMedia('(prefers-color-scheme: dark)'); } catch { return undefined; }
    const onChange = (e) => setSystemDark(e.matches);
    mq.addEventListener?.('change', onChange);
    return () => mq.removeEventListener?.('change', onChange);
  }, []);

  const isDark = theme === 'system' ? systemDark : theme === 'dark';

  // The top-bar switch picks an explicit theme (leaving "system" mode).
  const toggleTheme = () => setPreference('theme', isDark ? 'light' : 'dark');

  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', !isDark);
    document.documentElement.classList.toggle('dark', isDark);
  }, [isDark]);

  return (
    <ThemeContext.Provider value={{ isDark, theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
