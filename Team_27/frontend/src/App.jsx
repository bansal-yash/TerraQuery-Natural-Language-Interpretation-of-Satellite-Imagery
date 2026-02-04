import React, { useState, useRef, useEffect, useCallback } from 'react';

// ============================================================================
// ASSETS & ICONS
// ============================================================================

const IsroLogo = ({ collapsed }) => (
  <div className={`flex items-center gap-4 transition-all ${collapsed ? 'justify-center' : ''}`}>
    <div className="relative w-12 h-12 shrink-0 flex items-center justify-center bg-[#0E1015] rounded-xl shadow-[0_0_20px_rgba(59,130,246,0.15)] overflow-hidden border border-blue-500/30 group-hover:border-blue-400 transition-colors">
      <img 
        src="./TerraQuery.png" 
        alt="TerraQuery" 
        className="w-full h-full object-cover scale-95"
        onError={(e) => {
          e.target.style.display = 'none';
          e.target.parentNode.innerHTML = '<div class="w-full h-full bg-blue-900/30 flex items-center justify-center"><span class="text-[10px] font-bold text-blue-200">TQ</span></div>';
        }}
      />
    </div>
    {!collapsed && (
      <div className="flex flex-col overflow-hidden whitespace-nowrap">
        <h1 className="text-xl font-black tracking-wider text-white leading-none truncate font-sans">
          TERRA<span className="text-orange-500 font-light">QUERY</span>
        </h1>
        <span className="text-[8px] font-bold text-blue-300/70 tracking-[0.2em] uppercase mt-1.5 truncate flex items-center gap-1.5">
          <span className="w-1 h-1 rounded-full bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.8)] animate-pulse"></span>
          POWERED BY GEONLI AI
        </span>
      </div>
    )}
  </div>
);

const Icons = {
  Upload: () => <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>,
  Send: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>,
  Satellite: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  RotateRight: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
  Flip: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" /></svg>,
  ZoomIn: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" /></svg>,
  ZoomOut: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" /></svg>,
  Crop: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12L9.121 9.121m0 5.758a3 3 0 10-4.243 4.243 3 3 0 004.243-4.243zm0-5.758a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z" /></svg>,
  Check: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>,
  Cancel: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>,
  Close: () => <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>,
  Dimensions: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>,
  ChevronDown: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>,
  ChevronUp: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" /></svg>,
  Brain: () => <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>,
  Menu: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>,
  Plus: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>,
  Trash: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>,
  Save: () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" /></svg>
};

const API_URL = "/api"; 
const DEFAULT_CAPTION_PROMPT = "Provide a concise satellite imagery caption highlighting key geography, activity, and risk cues.";

const extractErrorMessage = async (response) => {
  if (!response) return 'API Error';
  try {
    const raw = await response.text();
    if (!raw) return 'API Error';
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed === 'string') return parsed;
      return parsed.detail || parsed.message || parsed.error || 'API Error';
    } catch {
      return raw.trim() || 'API Error';
    }
  } catch {
    return 'API Error';
  }
};

// Defaults for layout reset on reload
const DEFAULT_SIDEBAR_WIDTH = 320;
const DEFAULT_CHAT_WIDTH = 384;
const DEFAULT_REPO_HEIGHT = 210;

// ============================================================================
// MAIN APPLICATION
// ============================================================================
function App() {
  // 1. Initialize History from LocalStorage (Persistent)
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem('geonli_sessions');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.warn("Failed to load local history", e);
      return [];
    }
  });

  // 2. Initialize Current Session - Always start fresh on reload
  const createNewSession = () => ({
    id: null,
    title: "MISSION_START",
    startTime: Date.now(),
    imageUrl: null,
    activeLabel: "OG", // Tracks the label of the currently displayed image for lineage
    imageDims: { w: 0, h: 0 },
    messages: [],
    currentCaption: "Awaiting satellite imagery stream...",
    groundedImages: [] 
  });

  // Always start with a fresh session on page load (history is preserved in sessions)
  const [currentSession, setCurrentSession] = useState(createNewSession);

  // 3. UI State (Resets on reload)
  const [transform, setTransform] = useState({ rotate: 0, scale: 1, flipX: 1, translateX: 0, translateY: 0 });
  const [isCropping, setIsCropping] = useState(false);
  const [selection, setSelection] = useState(null); 
  const [dragStart, setDragStart] = useState(null);
  const [viewImage, setViewImage] = useState(null);
  const [expandedThoughts, setExpandedThoughts] = useState({});

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [autoCaptionSessionId, setAutoCaptionSessionId] = useState(null);

  // Layout State
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [chatPanelWidth, setChatPanelWidth] = useState(DEFAULT_CHAT_WIDTH); 
  const [repoPanelHeight, setRepoPanelHeight] = useState(DEFAULT_REPO_HEIGHT); 
  
  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  
  // Resize refs
  const isResizingChat = useRef(false);
  const isResizingRepo = useRef(false);
  const isResizingSidebar = useRef(false);

  // --- BROWSER FAVICON & TITLE LOGIC ---
  useEffect(() => {
    // Dynamically set favicon to our logo
    const existingLink = document.querySelector("link[rel~='icon']");
    if (existingLink) {
        existingLink.href = './TerraQuery.png';
    } else {
        const link = document.createElement('link');
        link.type = 'image/png';
        link.rel = 'icon';
        link.href = './TerraQuery.png';
        document.getElementsByTagName('head')[0].appendChild(link);
    }
    // Set Page Title
    document.title = "TerraQuery | GeoNLI";
  }, []);

  // --- PERSISTENCE EFFECT ---
  // --- PERSISTENCE EFFECT ---
  useEffect(() => {
    try {
      let updatedSessions = [...sessions];
      if (currentSession.id) {
         const idx = updatedSessions.findIndex(s => s.id === currentSession.id);
         if (idx >= 0) {
            updatedSessions[idx] = currentSession;
         } else {
            updatedSessions = [currentSession, ...updatedSessions];
         }
      }
      localStorage.setItem('geonli_sessions', JSON.stringify(updatedSessions));
    } catch (e) {
      console.error("Storage Limit Exceeded", e);
    }
  }, [sessions, currentSession]);

  // Save current session to history before page unload (reload/close)
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (currentSession.id) {
        try {
          const savedSessions = localStorage.getItem('geonli_sessions');
          let updatedSessions = savedSessions ? JSON.parse(savedSessions) : [];
          const idx = updatedSessions.findIndex(s => s.id === currentSession.id);
          if (idx >= 0) {
            updatedSessions[idx] = currentSession;
          } else {
            updatedSessions = [currentSession, ...updatedSessions];
          }
          localStorage.setItem('geonli_sessions', JSON.stringify(updatedSessions));
        } catch (e) {
          console.error("Failed to save session before unload", e);
        }
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [currentSession]);

  // Sync helper
  const syncSessionToHistory = (updatedSession) => {
    setSessions(prev => {
        const idx = prev.findIndex(s => s.id === updatedSession.id);
        if (idx >= 0) {
            const newSessions = [...prev];
            newSessions[idx] = updatedSession;
            return newSessions;
        }
        return [updatedSession, ...prev];
    });
  };

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [currentSession.messages, isLoading, isAnalyzing]);

  // --- RESIZING LOGIC ---
  const startResizeChat = useCallback(() => { isResizingChat.current = true; }, []);
  const startResizeRepo = useCallback(() => { isResizingRepo.current = true; }, []);
  const startResizeSidebar = useCallback(() => { isResizingSidebar.current = true; }, []);

  const handleResize = useCallback((e) => {
    if (isResizingChat.current) {
        const newWidth = window.innerWidth - e.clientX;
        if (newWidth > 250 && newWidth < 800) setChatPanelWidth(newWidth);
    }
    if (isResizingRepo.current) {
        const newHeight = window.innerHeight - e.clientY;
        if (newHeight > 0 && newHeight < window.innerHeight * 0.6) setRepoPanelHeight(newHeight);
    }
    if (isResizingSidebar.current) {
        const newWidth = e.clientX;
        if (newWidth > 200 && newWidth < 500) setSidebarWidth(newWidth);
    }
  }, []);

  const stopResize = useCallback(() => {
    isResizingChat.current = false;
    isResizingRepo.current = false;
    isResizingSidebar.current = false;
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', handleResize);
    window.addEventListener('mouseup', stopResize);
    return () => {
      window.removeEventListener('mousemove', handleResize);
      window.removeEventListener('mouseup', stopResize);
    };
  }, [handleResize, stopResize]);

  // --- IMAGE HELPER: Apply Transforms to Blob ---
  const getTransformedImageBlob = async () => {
      if (!currentSession.imageUrl) return null;
      return new Promise((resolve) => {
          const img = new Image();
          img.crossOrigin = "anonymous";
          img.src = currentSession.imageUrl;
          img.onload = () => {
              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');
              const { rotate, flipX } = transform;
              if (rotate % 180 !== 0) {
                  canvas.width = img.height;
                  canvas.height = img.width;
              } else {
                  canvas.width = img.width;
                  canvas.height = img.height;
              }
              ctx.translate(canvas.width / 2, canvas.height / 2);
              ctx.rotate((rotate * Math.PI) / 180);
              ctx.scale(flipX, 1);
              ctx.drawImage(img, -img.width / 2, -img.height / 2);
              canvas.toBlob((blob) => {
                  resolve(blob);
              }, 'image/jpeg', 0.95);
          };
          img.onerror = () => resolve(null);
      });
  };

  // 4. Smart Title Updater - generates 4-5 word summary from caption
  // Returns the new title so caller can use it immediately
  const generateTitleFromText = (text) => {
    const cleanText = text.replace(/(\r\n|\n|\r)/gm, " ").replace(/^(Analysis:|Error:)\s*/i, "");
    const words = cleanText.split(' ').filter(w => w.length > 0);
    const summary = words.slice(0, 5).join(' ') + (words.length > 5 ? "..." : "");
    return summary.length >= 3 ? summary : null;
  };

  const updateSessionTitleSmartly = (text, forceUpdate = false) => {
    setCurrentSession(prev => {
      // Check if title is still a default/temporary one that should be replaced
      const isDefaultTitle = 
        prev.title.startsWith("OP_") || 
        prev.title === "MISSION_START" || 
        prev.title.startsWith("TEMP-") ||
        prev.title === "Analysis Complete." ||
        prev.title.startsWith("Analysis:") ||
        forceUpdate;
        
      if (isDefaultTitle) {
        const newTitle = generateTitleFromText(text);
        if (newTitle) {
          const updatedSession = { ...prev, title: newTitle };
          // Sync to history immediately
          setTimeout(() => syncSessionToHistory(updatedSession), 0);
          return updatedSession;
        }
      }
      return prev;
    });
  };

  const sendCaptionRequest = async ({ promptText, isAutoPrompt = false, silent = false } = {}) => {
    if (!currentSession.imageUrl) return;
    const trimmedPrompt = (promptText || '').trim();
    if (!trimmedPrompt) return;

    let updatedMessages = currentSession.messages;
    if (!silent) {
        const displayPrompt = isAutoPrompt ? `[AUTO-CAPTION] ${trimmedPrompt}` : trimmedPrompt;
        const promptMessage = { role: 'user', text: displayPrompt };
        updatedMessages = [...updatedMessages, promptMessage];
        setCurrentSession(prev => {
            const next = { ...prev, messages: updatedMessages };
            syncSessionToHistory(next);
            return next;
        });
    }

    setIsLoading(true);
    const contextHistory = updatedMessages
      .map(m => {
          if (m.type === 'image') return `System: [Image Uploaded: ${m.label}]`;
          return `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.text}`
      })
      .join('\n') + (silent ? `\nUser: ${trimmedPrompt}` : '');

    try {
      const imageBlob = await getTransformedImageBlob();
      if (!imageBlob) throw new Error('Unable to prepare imagery payload.');

      const formData = new FormData();
      formData.append('image', imageBlob, 'image.jpg');
      formData.append('prompt', trimmedPrompt);
      formData.append('context', contextHistory);

      const response = await fetch(`${API_URL}/handle`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorMessage = await extractErrorMessage(response);
        throw new Error(errorMessage);
      }

      const data = await response.json();
      if (data.routed_to !== 'CAPTION') {
        throw new Error('Only caption responses are supported in this mode.');
      }

      const responseText = data.result?.answer || 'No caption generated.';
      
      // Generate title from caption immediately
      const newTitle = generateTitleFromText(responseText);

      const assistantMessage = {
        role: 'assistant',
        text: responseText,
        routeType: 'CAPTION',
        thinking: data.result?.thinking || null
      };

      // Build final session with title updated if needed
      setCurrentSession(prev => {
        const shouldUpdateTitle = prev.title.startsWith("OP_") || 
          prev.title === "MISSION_START" || 
          prev.title.startsWith("TEMP-") ||
          isAutoPrompt || silent;
        
        const finalSession = {
          ...prev,
          title: (shouldUpdateTitle && newTitle) ? newTitle : prev.title,
          messages: silent ? [...prev.messages, assistantMessage] : [...updatedMessages, assistantMessage],
          currentCaption: responseText
        };
        
        // Sync to history
        setTimeout(() => syncSessionToHistory(finalSession), 0);
        return finalSession;
      });
    } catch (err) {
      setCurrentSession(prev => ({
        ...prev,
        messages: [...prev.messages, {
          role: 'assistant',
          text: `Error: ${err.message || 'Backend unavailable. Please try again.'}`
        }]
      }));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!autoCaptionSessionId) return;
    if (autoCaptionSessionId !== currentSession.id) return;
    if (!currentSession.imageUrl) return;
    sendCaptionRequest({ promptText: DEFAULT_CAPTION_PROMPT, isAutoPrompt: true, silent: true });
    setAutoCaptionSessionId(null);
  }, [autoCaptionSessionId, currentSession.id, currentSession.imageUrl]);

  // --- HISTORY MANAGEMENT ---
  const handleNewSession = () => {
    if (currentSession.id) syncSessionToHistory(currentSession);
    const newSession = createNewSession();
    setCurrentSession(newSession);
    setTransform({ rotate: 0, scale: 1, flipX: 1, translateX: 0, translateY: 0 });
    setIsCropping(false);
    setSelection(null);
    setAutoCaptionSessionId(null);
    // Clear active session ID since we're starting fresh
    localStorage.removeItem('geonli_active_session_id');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const loadHistorySession = (session) => {
    if (currentSession.id === session.id) return;
    if (currentSession.id) syncSessionToHistory(currentSession);
    setCurrentSession(session);
    setTransform({ rotate: 0, scale: 1, flipX: 1, translateX: 0, translateY: 0 });
    setIsCropping(false);
    setAutoCaptionSessionId(null);
    // Update active session ID when switching sessions
    if (session.id) {
      localStorage.setItem('geonli_active_session_id', session.id);
    }
  };

  const handleDeleteSession = (e, sessionId) => {
    e.stopPropagation();
    if (window.confirm("Are you sure you want to delete this mission log? This action cannot be undone.")) {
        setSessions(prev => {
            const updated = prev.filter(s => s.id !== sessionId);
            // If deleting current session, reset to new
            if (currentSession.id === sessionId) {
                // We use setTimeout to break the render cycle and avoid state clashes
                setTimeout(() => handleNewSession(), 0);
            }
            // Update storage immediately
            localStorage.setItem('geonli_sessions', JSON.stringify(updated));
            return updated;
        });
    }
  };

  // --- IMAGE OPERATIONS ---
  const handleRotate = () => setTransform(prev => ({ ...prev, rotate: (prev.rotate + 90) % 360 }));
  const handleFlip = () => setTransform(prev => ({ ...prev, flipX: prev.flipX === 1 ? -1 : 1 }));
  const handleZoom = (delta) => setTransform(prev => ({ ...prev, scale: Math.max(0.5, Math.min(5, prev.scale + delta)) }));
  
  const handleResetTransform = () => setTransform({ rotate: 0, scale: 1, flipX: 1, translateX: 0, translateY: 0 });
  const activateCropMode = () => { handleResetTransform(); setIsCropping(true); setSelection(null); };

  // New Helper: Generate Version Label based on parent and action
  const generateNextLabel = (parentLabel, action) => {
      // Create a set of existing labels to check against
      const existing = currentSession.groundedImages.map(i => i.label);
      
      if (action === 'crop') {
          // If parent is a descendant of CROP
          const cropMatch = parentLabel.match(/^(CROP [\d\.]+)/);
          
          if (cropMatch) {
             const base = cropMatch[1]; // e.g. "CROP 1" or "CROP 2.1"
             const escapeBase = base.replace('.', '\\.');
             const childRegex = new RegExp(`^${escapeBase}\\.(\\d+)`);
             
             const childrenIndices = existing
                .map(l => {
                    const m = l.match(childRegex);
                    return m ? parseInt(m[1]) : 0;
                });
             const maxChild = Math.max(0, ...childrenIndices);
             return `${base}.${maxChild + 1}`;
          }
          
          // Otherwise (OG, OG FLIP, TRANSFORMED, etc.), treat as top-level crop
          // Find max "CROP N"
          const numbers = existing
            .map(l => {
                const m = l.match(/^CROP (\d+)/);
                return m ? parseInt(m[1]) : 0;
            });
          const maxNum = Math.max(0, ...numbers);
          return `CROP ${maxNum + 1}`;
      }
      
      if (action === 'transform') {
         // Simply append the action to the current label
         let suffix = "";
         if (transform.rotate !== 0) suffix += ` ROTATE ${Math.floor(Math.abs(transform.rotate) / 90)}`;
         if (transform.flipX !== 1) suffix += ` FLIP 1`;
         return `${parentLabel}${suffix}`;
      }
      
      return `IMG_${Date.now()}`;
  };

  const handleSaveTransform = async () => {
      if (transform.rotate === 0 && transform.flipX === 1) return; // Nothing to save
      
      const blob = await getTransformedImageBlob();
      if (!blob) return;
      
      // Convert blob to base64 for local display
      const reader = new FileReader();
      reader.onloadend = () => {
          const transformedUrl = reader.result;
          const newLabel = generateNextLabel(currentSession.activeLabel, 'transform');
          
          const newImageEntry = { 
              id: Date.now(), 
              url: transformedUrl, 
              label: newLabel, 
              coords: "TRANSFORMED", 
              type: 'source' 
          };
          
          // UPDATED: Do NOT add message to chat, only to Gallery (groundedImages)
          const updatedSession = {
              ...currentSession,
              imageUrl: transformedUrl,
              activeLabel: newLabel,
              groundedImages: [newImageEntry, ...currentSession.groundedImages]
          };

          setCurrentSession(updatedSession);
          syncSessionToHistory(updatedSession);
          // Reset visual transforms since the image source itself is now transformed
          handleResetTransform();
      };
      reader.readAsDataURL(blob);
  };

  const performCrop = () => {
    if (!imageRef.current || !selection || selection.width < 5 || selection.height < 5) return;
    const canvas = document.createElement('canvas');
    const scaleX = imageRef.current.naturalWidth / imageRef.current.clientWidth;
    const scaleY = imageRef.current.naturalHeight / imageRef.current.clientHeight;
    
    const imageRect = imageRef.current.getBoundingClientRect();
    const containerRect = containerRef.current.getBoundingClientRect();
    const offsetX = imageRect.left - containerRect.left;
    const offsetY = imageRect.top - containerRect.top;

    const pixelX = (selection.x - offsetX) * scaleX;
    const pixelY = (selection.y - offsetY) * scaleY;
    const pixelWidth = selection.width * scaleX;
    const pixelHeight = selection.height * scaleY;

    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageRef.current, pixelX, pixelY, pixelWidth, pixelHeight, 0, 0, pixelWidth, pixelHeight);
    
    const croppedImageUrl = canvas.toDataURL('image/jpeg');
    const newLabel = generateNextLabel(currentSession.activeLabel, 'crop');
    
    // Add to repo
    const newImageEntry = { 
        id: Date.now(), 
        url: croppedImageUrl, 
        label: newLabel, 
        coords: "CROPPED", 
        type: 'source' 
    };

    // UPDATED: Do NOT add message to chat, only to Gallery (groundedImages)
    const updatedSession = {
      ...currentSession,
      imageUrl: croppedImageUrl,
      activeLabel: newLabel,
      imageDims: { w: Math.round(pixelWidth), h: Math.round(pixelHeight) },
      groundedImages: [newImageEntry, ...currentSession.groundedImages],
    };
    
    setCurrentSession(updatedSession);
    syncSessionToHistory(updatedSession);
    setIsCropping(false);
    setSelection(null);
    handleResetTransform(); // Reset any pending transforms after crop
  };

  const handleMouseDown = (e) => {
    if (!isCropping || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDragStart({ x, y });
    setSelection({ x, y, width: 0, height: 0 });
  };
  const handleMouseMove = (e) => {
    if (!isCropping || !dragStart || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    setSelection({ x: Math.min(dragStart.x, currentX), y: Math.min(dragStart.y, currentY), width: Math.abs(currentX - dragStart.x), height: Math.abs(currentY - dragStart.y) });
  };
  const handleMouseUp = () => setDragStart(null);

  // --- API HANDLERS ---
  const handleFile = async (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
      const img = new Image();
      img.src = e.target.result;
      img.onload = async () => {
        if (currentSession.id) syncSessionToHistory(currentSession);

        const tempId = `OP_${Date.now().toString().slice(-6)}`;
        
        // Initial "OG" image entry
        const ogEntry = {
             id: Date.now(),
             url: e.target.result,
             label: "OG",
             coords: "UPLOAD",
             type: 'source'
        };

        // Initial Message showing the image (Only OG is shown in Chat initially)
        const initImageMsg = {
            role: 'system',
            type: 'image',
            url: e.target.result,
            label: "OG",
            text: "Original Image Uploaded"
        };

        const initSession = { 
          id: tempId,
          imageUrl: e.target.result,
          activeLabel: "OG",
          imageDims: { w: img.width, h: img.height }, 
          title: tempId,
          currentCaption: "Establishing connection...", 
          messages: [initImageMsg],
          groundedImages: [ogEntry] 
        };
        
        setCurrentSession(initSession);
        setTransform({ rotate: 0, scale: 1, flipX: 1, translateX: 0, translateY: 0 });
        setIsAnalyzing(true); 

        const formData = new FormData();
        formData.append('image', file);

        setAutoCaptionSessionId(null);

        try {
          const response = await fetch(`${API_URL}/api/new-session`, { method: 'POST', body: formData });
          if (!response.ok) throw new Error("API_ERROR");
          const data = await response.json();
          const safeId = data.session_id || tempId;
          const safeCaption = data.response || "Analysis Complete.";
          
          // Use the first caption for the Title immediately (4-5 word summary)
          const cleanCaption = safeCaption.replace(/(\r\n|\n|\r)/gm, " ");
          const words = cleanCaption.split(' ').filter(w => w.length > 0);
          const titleSummary = words.slice(0, 5).join(' ') + (words.length > 5 ? "..." : "");

          const readySession = {
            ...initSession,
            id: safeId,
            title: titleSummary, // Set title from initial caption (4-5 words)
            currentCaption: safeCaption,
            messages: [...initSession.messages, { role: 'assistant', text: `Analysis: ${safeCaption}` }]
          };
          setCurrentSession(readySession);
          syncSessionToHistory(readySession);
          setAutoCaptionSessionId(safeId);
        } catch (err) {
          console.warn("Backend unavailable, switching to local caption mode.");
          const errorSession = {
            ...initSession,
            currentCaption: "Satellite feed received.",
            messages: [...initSession.messages, { role: 'assistant', text: "Satellite feed received. Attempting caption service." }]
          };
          setCurrentSession(errorSession);
          setAutoCaptionSessionId(tempId);
        } finally {
          setIsAnalyzing(false);
        }
      };
    };
    reader.readAsDataURL(file);
  };

  const handleSend = async () => {
    if (!input.trim() || !currentSession.imageUrl) return;
    const userText = input;
    setInput('');
    
    const updatedMessages = [...currentSession.messages, { role: 'user', text: userText }];
    const optimisticSession = { ...currentSession, messages: updatedMessages };
    setCurrentSession(optimisticSession);
    syncSessionToHistory(optimisticSession);
    
    setIsLoading(true);

    const contextHistory = updatedMessages
        .map(m => {
          if (m.type === 'image') return `System: [Image Uploaded: ${m.label}]`;
          return `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.text}`
        })
        .join('\n');

    try {
      const imageBlob = await getTransformedImageBlob();
      
      const formData = new FormData();
      formData.append('image', imageBlob || new Blob(), 'image.jpg');
      formData.append('prompt', userText);
      formData.append('context', contextHistory);

      const response = await fetch(`${API_URL}/handle`, { 
        method: 'POST', 
        body: formData 
      });

      if (!response.ok) {
        const errorMessage = await extractErrorMessage(response);
        throw new Error(errorMessage);
      }

      const data = await response.json();
      const { routed_to, result } = data;

      let responseText = '';
      let thinking = result.thinking || null;
      let imageWithBoxes = result.image_with_boxes || null;

      if (routed_to === 'CAPTION') {
        responseText = result.answer || 'No caption generated.';
      } else if (routed_to === 'GROUND') {
        responseText = result.answer || 'Grounding complete.';
        if (imageWithBoxes) {
          const originalEntry = {
            id: Date.now() - 1,
            url: currentSession.imageUrl,
            label: currentSession.activeLabel, // The input image label
            coords: 'Source',
            type: 'source'
          };
          const groundingEntry = { 
            id: Date.now(), 
            url: imageWithBoxes, 
            label: `BBOX: ${result.labels?.[0]?.split(' ')[0] || 'Objects'}`, 
            coords: `${result.boxes?.length || 0} detected`, 
            type: 'grounding' 
          };
          
          // Generate title from response
          const newTitle = generateTitleFromText(responseText);

          // Add grounding image to chat
          const groundingMsg = {
              role: 'assistant',
              type: 'image',
              url: imageWithBoxes,
              label: 'GROUNDING',
              text: responseText,
              thinking: thinking
          };

          setCurrentSession(prev => {
            const shouldUpdateTitle = prev.title.startsWith("OP_") || 
              prev.title === "MISSION_START" || 
              prev.title.startsWith("TEMP-");
            
            const groundedSession = {
              ...prev,
              title: (shouldUpdateTitle && newTitle) ? newTitle : prev.title,
              currentCaption: responseText,
              groundedImages: [groundingEntry, originalEntry, ...prev.groundedImages],
              messages: [...updatedMessages, groundingMsg]
            };
            
            setTimeout(() => syncSessionToHistory(groundedSession), 0);
            return groundedSession;
          });
          setIsLoading(false);
          return; 
        }
      } else if (routed_to === 'VQA_NUMERICAL') {
        responseText = `Count: ${result.answer}`;
      } else if (routed_to === 'VQA_BINARY') {
        responseText = result.answer;
      } else if (routed_to === 'VQA_ATTRIBUTE') {
        responseText = result.answer;
      } else if (routed_to === 'VQA_FILTERING') {
        responseText = `Filtering complete for: ${result.answer}`;
        if (result.masks && result.masks.length > 0) {
          responseText += `\n\nFound ${result.masks.length} matching region(s):`;
          result.masks.forEach((mask, i) => {
            responseText += `\n• Region ${i + 1}: ${mask.reason || 'Match found'}`;
          });
        }
        if (imageWithBoxes) {
          // Generate title from response
          const filterTitle = generateTitleFromText(responseText);

          const filteredMsg = {
              role: 'assistant',
              type: 'image',
              url: imageWithBoxes,
              label: 'FILTER',
              text: responseText,
              thinking: thinking
          };

          setCurrentSession(prev => {
            const shouldUpdateTitle = prev.title.startsWith("OP_") || 
              prev.title === "MISSION_START" || 
              prev.title.startsWith("TEMP-");
            
            const filteredSession = {
              ...prev,
              title: (shouldUpdateTitle && filterTitle) ? filterTitle : prev.title,
              currentCaption: responseText,
              groundedImages: prev.groundedImages,
              messages: [...updatedMessages, filteredMsg]
            };
            
            setTimeout(() => syncSessionToHistory(filteredSession), 0);
            return filteredSession;
          });
          setIsLoading(false);
          return;
        }
      } else {
        responseText = result.answer || 'Some error occurred.';
      }

      // Update title if still a default one
      const newTitle = generateTitleFromText(responseText);

      setCurrentSession(prev => {
        const shouldUpdateTitle = prev.title.startsWith("OP_") || 
          prev.title === "MISSION_START" || 
          prev.title.startsWith("TEMP-");
        
        const finalGenericSession = {
          ...prev,
          title: (shouldUpdateTitle && newTitle) ? newTitle : prev.title,
          currentCaption: responseText,
          messages: [...updatedMessages, { 
            role: 'assistant', 
            text: responseText, 
            routeType: routed_to,
            thinking: thinking 
          }]
        };
        
        setTimeout(() => syncSessionToHistory(finalGenericSession), 0);
        return finalGenericSession;
      });

    } catch (err) {
      console.error('API Error:', err);
      setCurrentSession(prev => ({ 
        ...prev, 
        messages: [...updatedMessages, { 
          role: 'assistant', 
          text: `Error: ${err.message || 'Backend unavailable. Please try again.'}` 
        }] 
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const toggleThought = (idx) => {
    setExpandedThoughts(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const handleRepoClick = (img) => {
    if (img.type === 'grounding') setViewImage(img);
    else {
      // Just restore the view
      const restoredSession = { 
          ...currentSession, 
          imageUrl: img.url, 
          activeLabel: img.label // Set active label to the one clicked
      };
      setCurrentSession(restoredSession);
      syncSessionToHistory(restoredSession);
      handleResetTransform();
    }
  };

  // UI VARS
  const isBusy = isLoading || isAnalyzing;
  const isTransforming = transform.rotate !== 0 || transform.flipX !== 1;

  return (
    <div className="flex h-screen w-screen bg-[#050508] text-gray-100 font-sans overflow-hidden selection:bg-orange-500/30">
      
      {/* LIGHTBOX */}
      {viewImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8 animate-in fade-in duration-200" onClick={() => setViewImage(null)}>
          <div className="relative max-w-5xl max-h-full bg-[#0A0A0F] border border-white/10 rounded-xl shadow-2xl p-2 flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center px-4 py-2 border-b border-white/5 bg-[#0F1016] rounded-t-lg"><div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div><span className="text-sm font-bold text-blue-100 tracking-widest">{viewImage.label}</span></div><button onClick={() => setViewImage(null)} className="text-gray-400 hover:text-white"><Icons.Close /></button></div>
            <div className="flex-1 overflow-auto p-4 flex items-center justify-center"><img src={viewImage.url} alt="Grounded View" className="max-w-full max-h-[80vh] object-contain rounded-lg border border-blue-500/30 shadow-[0_0_30px_rgba(59,130,246,0.1)]" /></div>
          </div>
        </div>
      )}

      {/* LEFT SIDEBAR */}
      <div 
        style={{ width: isSidebarCollapsed ? 80 : sidebarWidth }} 
        className="bg-[#0A0A0F] border-r border-white/5 flex flex-col z-20 shadow-[4px_0_24px_rgba(0,0,0,0.4)] transition-width duration-200 relative shrink-0"
      >
        <div className={`p-4 border-b border-white/5 bg-gradient-to-r from-blue-900/10 to-transparent flex items-center ${isSidebarCollapsed ? 'justify-center' : 'justify-between'}`}>
            <div onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)} className="cursor-pointer transition-opacity hover:opacity-80">
                <IsroLogo collapsed={isSidebarCollapsed} />
            </div>
             
             <button 
                onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)} 
                className={`text-gray-400 hover:text-white transition-colors p-1 absolute right-2 top-6 ${isSidebarCollapsed ? 'hidden' : ''}`}
            >
               <Icons.Menu />
            </button>
        </div>

        {!isSidebarCollapsed && (
            <>
        <div className="p-6 pb-2">
            <button 
                onClick={handleNewSession} 
                disabled={isBusy}
                className={`w-full relative group overflow-hidden rounded-md bg-gradient-to-r from-orange-600 to-orange-500 p-[1px] shadow-lg shadow-orange-900/20 transition-all ${isBusy ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-orange-500/20'}`}
            >
                <div className="relative bg-[#0A0A0F] group-hover:bg-opacity-0 transition-colors duration-300 rounded-[5px] py-3 flex items-center justify-center gap-3">
                    <span className="text-sm font-bold tracking-widest uppercase text-orange-500 group-hover:text-white transition-colors">+ New Operation</span>
                </div>
            </button>
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-4 custom-scrollbar">
            <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-[0.2em] mb-4 pl-2">Operation Logs</h3>
            <div className="space-y-1">
                {sessions.map((session, i) => (
                    <div key={session.id || i} onClick={() => loadHistorySession(session)} className={`group relative p-3 rounded-md cursor-pointer border-l-2 transition-all duration-200 ${currentSession.id === session.id ? 'bg-blue-500/10 border-blue-400' : 'bg-transparent border-transparent hover:bg-white/5 hover:border-gray-600'}`}>
                        <div className={`font-mono text-xs font-medium truncate uppercase tracking-wide pr-6 ${currentSession.id === session.id ? 'text-blue-200' : 'text-gray-400 group-hover:text-gray-200'}`}>{session.title}</div>
                        <button 
                            onClick={(e) => handleDeleteSession(e, session.id)} 
                            className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 text-red-500/50 hover:text-red-400 hover:bg-red-500/10 p-1.5 rounded-md transition-all"
                            title="Delete Log"
                        >
                            <Icons.Trash />
                        </button>
                    </div>
                ))}
            </div>
        </div>
            </>
        )}
        {isSidebarCollapsed && (
             <div className="flex-1 flex flex-col items-center py-4 gap-4">
                 <button onClick={handleNewSession} disabled={isBusy} className={`p-3 rounded-md bg-gradient-to-r from-orange-600 to-orange-500 text-white shadow-lg transition-all ${isBusy ? 'opacity-50' : 'hover:shadow-orange-500/20'}`} title="New Operation">
                     <Icons.Plus />
                 </button>
                 {sessions.map((session, i) => (
                    <div key={session.id || i} onClick={() => loadHistorySession(session)} className={`w-10 h-10 rounded-full flex items-center justify-center cursor-pointer border-2 transition-all duration-200 ${currentSession.id === session.id ? 'bg-blue-500/20 border-blue-400 text-blue-200' : 'bg-transparent border-gray-700 text-gray-400 hover:border-gray-500'}`} title={session.title}>
                        <span className="text-[10px] font-bold">{i + 1}</span>
                    </div>
                 ))}
             </div>
        )}
        
        {!isSidebarCollapsed && (
            <div 
                onMouseDown={startResizeSidebar}
                className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500/50 transition-colors z-30"
            />
        )}
      </div>

      {/* CENTER STAGE */}
      <div className="flex-1 flex flex-col relative bg-[#050508] overflow-hidden">
        {/* Header with Dimensions and ZOOM TOOLBAR */}
        <div className="h-14 border-b border-white/5 bg-[#0A0A0F]/80 backdrop-blur-md flex items-center justify-between px-6 z-10 shrink-0">
          <div className="flex items-center gap-4">
              <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2"><Icons.Dimensions /><span>Input Dimensions:</span></div>
              <div className="font-mono text-sm text-blue-400">{currentSession.imageUrl ? `${currentSession.imageDims.w} x ${currentSession.imageDims.h} px` : "NO DATA"}</div>
          </div>
           
          {/* TOOLBAR */}
          {currentSession.imageUrl && (
            <div className="bg-[#0A0A0F]/90 backdrop-blur-md border border-white/10 rounded-full px-3 py-1.5 flex items-center gap-3 shadow-sm">
                {isCropping ? (
                <><span className="text-xs text-orange-500 font-bold uppercase animate-pulse">Select Area</span><div className="w-[1px] h-4 bg-white/10"></div><button onClick={performCrop} disabled={!selection} className="text-green-400 hover:text-green-300 disabled:opacity-30"><Icons.Check /></button><button onClick={() => { setIsCropping(false); setSelection(null); }} className="text-red-400 hover:text-red-300"><Icons.Cancel /></button></>
                ) : (
                <>
                <button onClick={handleRotate} className="hover:text-blue-400 transition-colors"><Icons.RotateRight /></button>
                <button onClick={handleFlip} className="hover:text-blue-400 transition-colors"><Icons.Flip /></button>
                {/* TRANSFORM SAVE BUTTON */}
                <button 
                    onClick={handleSaveTransform} 
                    disabled={!isTransforming}
                    className={`transition-colors ${isTransforming ? 'text-green-400 hover:text-green-300' : 'text-gray-600 cursor-not-allowed'}`}
                    title="Save Transform"
                >
                    <Icons.Save />
                </button>
                <div className="w-[1px] h-4 bg-white/10"></div>
                <button onClick={activateCropMode} className="hover:text-orange-400 transition-colors"><Icons.Crop /></button>
                <div className="w-[1px] h-4 bg-white/10"></div>
                <button onClick={() => handleZoom(0.2)} className="hover:text-blue-400 transition-colors"><Icons.ZoomIn /></button>
                <button onClick={() => handleZoom(-0.2)} className="hover:text-blue-400 transition-colors"><Icons.ZoomOut /></button>
                </>
                )}
            </div>
           )}
        </div>

        <div className="flex-1 flex flex-col relative overflow-hidden">
          {/* Main Image Canvas */}
          <div className="flex-1 flex items-center justify-center p-8 relative overflow-hidden bg-black/20" onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
            {currentSession.imageUrl ? (
              <div className="relative group flex items-center justify-center max-w-full max-h-full" ref={containerRef}>
                <div style={{ transform: isCropping ? 'none' : `rotate(${transform.rotate}deg) scale(${transform.scale}) scaleX(${transform.flipX})` }} className="transition-transform duration-200 flex items-center justify-center">
                  <img src={currentSession.imageUrl} ref={imageRef} crossOrigin="anonymous" alt="Target" onLoad={(e) => setCurrentSession(prev => ({...prev, imageDims: { w: e.target.naturalWidth, h: e.target.naturalHeight }}))} className="max-h-[65vh] max-w-full w-auto h-auto object-contain rounded-sm shadow-2xl select-none pointer-events-none border border-white/5" />
                </div>
                {isCropping && selection && <div className="absolute border-2 border-orange-500 bg-orange-500/10 z-20 pointer-events-none" style={{ left: selection.x, top: selection.y, width: selection.width, height: selection.height }} />}
              </div>
            ) : (
              <div onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); }} className={`w-[500px] h-[300px] border border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-300 group cursor-pointer relative overflow-hidden ${isDragging ? 'border-orange-500 bg-orange-500/5' : 'border-gray-700 hover:border-gray-500 bg-[#0A0A0F]'}`} onClick={() => fileInputRef.current?.click()}><div className="absolute inset-0 bg-gradient-to-tr from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div><div className="p-4 bg-gray-800/50 rounded-full mb-4 border border-white/5"><div className="text-gray-400 group-hover:text-white"><Icons.Upload /></div></div><h3 className="text-lg font-medium text-gray-200">Upload Satellite Imagery</h3><input type="file" ref={fileInputRef} className="hidden" onChange={(e) => handleFile(e.target.files[0])} /></div>
            )}
          </div>

          {currentSession.imageUrl && (
            <>
            <div 
                onMouseDown={startResizeRepo}
                className="h-4 bg-[#050508] border-t border-b border-white/5 cursor-row-resize flex items-center justify-center hover:bg-blue-500/10 transition-colors z-30 group"
            >
                <div className="h-1 w-16 rounded-full bg-gray-700/50 group-hover:bg-blue-400/80 transition-colors shadow-sm"></div>
            </div>
            
            <div style={{ height: repoPanelHeight }} className="bg-[#08080C] flex flex-col z-20 shrink-0">
              <div className="px-6 py-2 border-b border-white/5 flex justify-between items-center bg-[#0A0A0F] shrink-0"><div className="flex items-center gap-2 text-[10px] font-bold text-blue-400 uppercase tracking-widest"><Icons.Satellite /><span>Image Gallery</span></div></div>
              <div className="flex-1 p-4 overflow-y-auto flex flex-wrap content-start gap-4 custom-scrollbar">
                {currentSession.groundedImages.map((img) => (
                  <div 
                    key={img.id} 
                    onClick={() => handleRepoClick(img)} 
                    className="w-36 aspect-square flex flex-col bg-[#0F1016] rounded-md border-2 overflow-hidden group cursor-pointer transition-all duration-300 hover:-translate-y-1 border-orange-500/40 hover:border-orange-500"
                  >
                    <div className="flex-1 relative overflow-hidden">
                      <img src={img.url} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" alt=""/>
                    </div>
                    <div className="p-2 shrink-0 bg-[#0A0A0F]/90 border-t border-white/5">
                      <div className={`text-[9px] font-bold truncate uppercase mb-0.5 ${img.type === 'grounding' ? 'text-blue-200' : 'text-orange-200'}`}>{img.label}</div>
                      <div className="text-[7px] font-mono text-gray-500 hover:text-white transition-colors">{img.type === 'grounding' ? 'Click to View' : 'Click to Restore'}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            </>
          )}
        </div>
      </div>

      <div 
        onMouseDown={startResizeChat}
        className="w-4 bg-[#050508] border-l border-r border-white/5 cursor-col-resize flex items-center justify-center z-30 hover:bg-blue-500/10 group transition-colors shadow-xl"
      >
        <div className="w-1 h-16 rounded-full bg-gray-700/50 group-hover:bg-blue-400/80 transition-colors shadow-sm"></div>
      </div>

      <div style={{ width: chatPanelWidth }} className="bg-[#0A0A0F] flex flex-col z-20 shrink-0">
        <div className="h-14 border-b border-white/5 flex items-center px-6 bg-[#08080C] shrink-0"><h2 className="text-xs font-bold text-gray-300 uppercase tracking-[0.15em]">Communication Link</h2><div className="ml-auto flex items-center gap-2"><span className="w-1.5 h-1.5 bg-emerald-500 rounded-full shadow-[0_0_8px_#10B981]"></span></div></div>
        <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar" ref={scrollRef}>
          {isAnalyzing ? (
             <div className="h-full flex flex-col items-center justify-center">
                <div className="w-16 h-16 rounded-full border-4 border-blue-500/30 border-t-blue-500 animate-spin mb-4"></div>
                <div className="text-xs font-mono text-blue-400 animate-pulse tracking-widest">SCANNING TOPOGRAPHY...</div>
             </div>
          ) : (
            currentSession.messages.map((msg, idx) => (
              <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
                <span className="text-[9px] text-gray-500 font-mono mb-1 uppercase tracking-wider px-1">
                  {msg.role === 'user' ? 'OPERATOR' : 'GeoNLI AI'}
                  {msg.routeType && <span className="ml-2 text-blue-400">[{msg.routeType}]</span>}
                </span>
                <div className={`max-w-[85%] rounded-lg text-xs leading-relaxed font-mono shadow-md ${msg.role === 'user' ? 'bg-blue-600/10 border border-blue-500/30 text-blue-100 rounded-br-none' : 'bg-gray-800/40 border border-gray-700/50 text-gray-300 rounded-bl-none'}`}>
                  {msg.type === 'image' ? (
                      <div className="p-1">
                          <img src={msg.url} alt={msg.label} className="w-full h-auto rounded-md border border-white/10" />
                          <div className="px-2 py-1 text-[10px] text-gray-400 font-bold uppercase tracking-wider flex justify-between items-center">
                              <span>{msg.label}</span>
                              {msg.text && <span className="text-[9px] text-gray-500 lowercase font-mono">{msg.text}</span>}
                          </div>
                          {/* If the message has additional text content (like VQA answers) */}
                          {msg.thinking && (
                            <div className="border-t border-gray-700/50 mt-1">
                              <button 
                                onClick={() => toggleThought(idx)}
                                className="w-full px-2 py-1.5 flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors"
                              >
                                <Icons.Brain />
                                <span className="text-[9px] font-semibold uppercase tracking-wider">Reasoning</span>
                                <span className="ml-auto">
                                  {expandedThoughts[idx] ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
                                </span>
                              </button>
                              {expandedThoughts[idx] && (
                                <div className="px-2 pb-2 pt-1 text-xs text-gray-400 bg-purple-500/5">
                                  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.thinking}</div>
                                </div>
                              )}
                            </div>
                          )}
                      </div>
                  ) : (
                      <>
                      {msg.thinking && (
                        <div className="border-b border-gray-700/50">
                          <button 
                            onClick={() => toggleThought(idx)}
                            className="w-full px-3 py-2 flex items-center gap-2 text-purple-400 hover:text-purple-300 hover:bg-purple-500/5 transition-colors"
                          >
                            <Icons.Brain />
                            <span className="text-[10px] font-semibold uppercase tracking-wider">Thought Process</span>
                            <span className="ml-auto">
                              {expandedThoughts[idx] ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
                            </span>
                          </button>
                          {expandedThoughts[idx] && (
                            <div className="px-3 pb-3 pt-1 text-sm text-gray-400 bg-purple-500/5 border-t border-purple-500/10">
                              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.thinking}</div>
                            </div>
                          )}
                        </div>
                      )}
                      <div className="p-3" style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
                      </>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && !isAnalyzing && (
            <div className="flex flex-col items-start animate-in fade-in slide-in-from-bottom-2 duration-300">
              <span className="text-[9px] text-gray-500 font-mono mb-1 uppercase tracking-wider px-1">GeoNLI AI</span>
              <div className="max-w-[85%] p-3 rounded-lg rounded-bl-none bg-gray-800/40 border border-gray-700/50">
                <div className="flex items-center gap-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-[10px] font-mono text-blue-400 animate-pulse">Thinking...</span>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="p-4 bg-[#08080C] border-t border-white/5 shrink-0"><div className="relative group"><input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !isLoading && handleSend()} placeholder="Enter query..." disabled={!currentSession.imageUrl || isLoading || isAnalyzing} className="w-full bg-[#12121A] text-gray-200 border border-white/5 rounded-lg pl-4 pr-12 py-3 text-xs font-mono focus:outline-none focus:border-blue-500/50 transition-all disabled:opacity-50" /><button onClick={handleSend} disabled={!currentSession.imageUrl || isLoading || isAnalyzing} className="absolute right-1 top-1 bottom-1 aspect-square flex items-center justify-center bg-blue-600/10 hover:bg-blue-600 text-blue-500 hover:text-white rounded-md"><Icons.Send /></button></div></div>
      </div>
    </div>
  );
}

export default App;