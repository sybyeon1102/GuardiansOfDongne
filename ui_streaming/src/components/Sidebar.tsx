// src/components/Sidebar.tsx

import { Calendar, Clock, Mail, Users, Globe, FileText, Settings } from "lucide-react";

export function Sidebar() {
  const icons = [
    { Icon: Calendar, active: false },
    { Icon: Clock, active: false },
    { Icon: Mail, active: false },
    { Icon: Users, active: false },
    { Icon: Globe, active: false },
    { Icon: FileText, active: false },
    { Icon: Settings, active: false },
  ];
  
  return (
    <aside className="w-16 bg-gray-200 flex flex-col items-center py-4 gap-4">
      {icons.map(({ Icon, active }, index) => (
        <button
          key={index}
          className={`w-10 h-10 flex items-center justify-center rounded-lg transition-colors ${
            active ? "bg-indigo-500 text-white" : "text-gray-600 hover:bg-gray-300"
          }`}
        >
          <Icon className="w-5 h-5" />
        </button>
      ))}
    </aside>
  );
}
