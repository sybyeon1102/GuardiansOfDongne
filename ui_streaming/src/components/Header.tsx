// src/components/Header.tsx

import { Search, Bell, User, Menu } from "lucide-react";

export function Header() {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="w-8 h-8 bg-indigo-500 rounded-lg flex items-center justify-center">
          <span className="text-white">G</span>
        </div>
        <h1 className="text-gray-800">GoD(Guardians of Dongne)</h1>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="relative flex items-center">
          <Menu className="absolute left-3 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search here"
            className="pl-10 pr-10 py-2 bg-gray-100 rounded-lg border-none outline-none w-80"
          />
          <Search className="absolute right-3 w-4 h-4 text-gray-400 cursor-pointer" />
        </div>
        
        <button className="p-2 hover:bg-gray-100 rounded-lg">
          <Bell className="w-5 h-5 text-gray-600" />
        </button>
        
        <button className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
          <User className="w-6 h-6 text-gray-600" />
        </button>
      </div>
    </header>
  );
}
