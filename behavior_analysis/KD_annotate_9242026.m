function annotate_disfluencies_video(videoPath)
% ANNOTATE_DISFLUENCIES_VIDEO  GUI for scoring stuttering / disfluency events from video.
%
%   annotate_disfluencies_video            opens an empty GUI (load a video via the button)
%   annotate_disfluencies_video(videoPath) opens the GUI and loads the given video
%
% Layout (top -> bottom): video | spectrogram | waveform | transcript panel |
% notes panel | button row. All event times are derived from the AUDIO sample
% clock, so temporal resolution is that of the audio track, not the video frames.
% During playback the video frame is also slaved to the audio player's sample
% position, so picture and sound stay in sync.
%
% Press "Shortcuts" in the GUI for the full key/mouse list.
%
% Requires only base MATLAB (no Signal Processing Toolbox). Audio is pulled from
% the video with AUDIOREAD (works from .mp4/.mov on Windows/macOS; on some Linux
% setups you may need to load a .wav container instead).
%
% All button/key/mouse callbacks are wrapped so any error pops up with the real
% message and the exact line number, instead of MATLAB's generic wrapper.

% ----------------------------------------------------------------------------
% Configurable event types (name, RGB colour, trigger key). Extend freely.
% ----------------------------------------------------------------------------
eventTypes = struct( ...
    'name',  {'repetition',  'block',      'prolongation'}, ...
    'color', {[0.85 0 0],    [0 0.2 0.9],  [0 0.6 0.1]},    ...
    'key',   {'r',           'b',          'p'});

% ----------------------------------------------------------------------------
% Shared state
% ----------------------------------------------------------------------------
MINVIEW = 0.02;
vid = [];  videoPath = ''; audio = []; fs = 0; dur = 0; audioMax = 1; maxFreq = 5000;
viewStart = 0; viewEnd = 1; cursorTime = 0; selStart = NaN; selEnd = NaN;
events = struct('start',{},'end',{},'type',{},'transcript',{},'notes',{});
currentEvent = [];

isEditing = false; editIdx = []; editPanel = '';
isPlaying = false; playStartTime = 0; playEndTime = 0;
dragAxes = []; dragStartT = 0; downPix = [0 0]; didDrag = false;

imgVideo = []; placeholderTxt = [];
hSpecImg = []; hWave = [];
evGfx = gobjects(0); selGfx = gobjects(0); curGfx = gobjects(0); playGfx = gobjects(0);

if nargin < 1, videoPath = ''; end

% ----------------------------------------------------------------------------
% Build figure
% ----------------------------------------------------------------------------
fig = figure('Name','Disfluency annotator','NumberTitle','off','Units','pixels', ...
    'Position',[80 60 1180 900],'Color',[0.94 0.94 0.94],'MenuBar','none', ...
    'Toolbar','none','WindowKeyPressFcn',@(~,e)cb(@()onKey(e)), ...
    'WindowScrollWheelFcn',@(~,e)cb(@()onScroll(e)),'CloseRequestFcn',@(~,~)onClose());

axVideo = axes('Parent',fig,'Units','normalized','Position',[0.06 0.53 0.88 0.44], ...
    'Color',[0 0 0]);
axis(axVideo,[0 1 0 1]); axis(axVideo,'off');
placeholderTxt = text(axVideo,0.5,0.5,'Load a video to begin', ...
    'HorizontalAlignment','center','FontSize',14,'Color',[0.6 0.6 0.6]);

axSpec  = axes('Parent',fig,'Units','normalized','Position',[0.08 0.375 0.86 0.135]);
axWave  = axes('Parent',fig,'Units','normalized','Position',[0.08 0.285 0.86 0.075]);
axTrans = axes('Parent',fig,'Units','normalized','Position',[0.08 0.205 0.86 0.065]);
axNotes = axes('Parent',fig,'Units','normalized','Position',[0.08 0.125 0.86 0.065]);

for ax = [axSpec axWave axTrans axNotes]
    hold(ax,'on'); box(ax,'on');
    set(ax,'XTick',[],'YTick',[],'Color',[1 1 1], ...
        'XColor',[0.15 0.15 0.15],'YColor',[0.15 0.15 0.15]);
end
set(axSpec,'YDir','normal'); ylabel(axSpec,'Freq (Hz)'); set(axSpec,'YTickMode','auto');
ylabel(axWave,'Amp');  ylim(axWave,[-1 1]);
ylabel(axTrans,'Transcript','Rotation',0,'HorizontalAlignment','right'); ylim(axTrans,[0 1]);
ylabel(axNotes,'Notes','Rotation',0,'HorizontalAlignment','right');      ylim(axNotes,[0 1]);
xlabel(axNotes,'Time (s)');

set(axSpec, 'ButtonDownFcn',@(~,~)cb(@()onAxDown(axSpec)));
set(axWave, 'ButtonDownFcn',@(~,~)cb(@()onAxDown(axWave)));
set(axTrans,'ButtonDownFcn',@(~,~)cb(@()onPanelDown(axTrans,'transcript')));
set(axNotes,'ButtonDownFcn',@(~,~)cb(@()onPanelDown(axNotes,'notes')));

editBox = uicontrol(fig,'Style','edit','Max',2,'Min',0,'Units','pixels', ...
    'HorizontalAlignment','left','FontSize',9,'Visible','off','Enable','off', ...
    'Callback',@(~,~)cb(@commitEdit));

% Buttons go through btnPress, which drops keyboard focus from the button so
% the space bar controls playback instead of re-clicking the last button.
uicontrol(fig,'Style','pushbutton','String','Load video','Units','normalized', ...
    'Position',[0.06 0.03 0.13 0.045],'Callback',@(src,~)cb(@()btnPress(src,@uiLoadVideo)));
uicontrol(fig,'Style','pushbutton','String','Save annotation','Units','normalized', ...
    'Position',[0.20 0.03 0.15 0.045],'Callback',@(src,~)cb(@()btnPress(src,@saveAnnotations)));
btnLoadAnnot = uicontrol(fig,'Style','pushbutton','String','Load annotation','Enable','off', ...
    'Units','normalized','Position',[0.36 0.03 0.15 0.045], ...
    'Callback',@(src,~)cb(@()btnPress(src,@loadAnnotations)));
uicontrol(fig,'Style','pushbutton','String','Shortcuts','Units','normalized', ...
    'Position',[0.52 0.03 0.11 0.045],'Callback',@(src,~)cb(@()btnPress(src,@showShortcuts)));

legStr = '';
for ti = 1:numel(eventTypes)
    legStr = [legStr sprintf('[%s] %s   ',upper(eventTypes(ti).key),eventTypes(ti).name)]; %#ok<AGROW>
end
uicontrol(fig,'Style','text','String',['Events:  ' legStr],'Units','normalized', ...
    'Position',[0.65 0.028 0.30 0.05],'HorizontalAlignment','left', ...
    'BackgroundColor',[0.94 0.94 0.94],'FontSize',9);

statusTxt = uicontrol(fig,'Style','text','String','No video loaded','Units','normalized', ...
    'Position',[0.06 0.08 0.88 0.03],'HorizontalAlignment','left', ...
    'BackgroundColor',[0.94 0.94 0.94],'FontSize',9,'ForegroundColor',[0.2 0.2 0.2]);

playTimer = timer('ExecutionMode','fixedRate','Period',0.03,'BusyMode','drop', ...
    'TimerFcn',@(~,~)cb(@playTick));

if ~isempty(videoPath), cb(@()loadVideo(videoPath)); end

% ============================================================================
%                               NESTED FUNCTIONS
% ============================================================================

    % ---------------- universal callback guard ------------------------------
    function cb(fn)
        try
            fn();
        catch err
            loc = '';
            % Report the first frame that lives in THIS file, not the deepest
            % built-in frame (e.g. uistack.m), so the reported line is useful.
            frame = [];
            for si = 1:numel(err.stack)
                if strcmp(err.stack(si).name,'annotate_disfluencies_video') || ...
                        contains(err.stack(si).name,'annotate_disfluencies_video/')
                    frame = err.stack(si); break;
                end
            end
            if isempty(frame) && ~isempty(err.stack), frame = err.stack(1); end
            if ~isempty(frame)
                loc = sprintf('\n\n(in %s, line %d)',frame.name,frame.line);
            end
            if exist('statusTxt','var') && ishghandle(statusTxt)
                set(statusTxt,'String',['ERROR: ' err.message]);
            end
            errordlg([err.message loc],'Error');
        end
    end

    % ---------------- button focus handling ---------------------------------
    function btnPress(src,fn)
        % Toggling Enable off/on removes keyboard focus from the button, so a
        % later space-bar press goes to WindowKeyPressFcn and not the button.
        try
            set(src,'Enable','off'); drawnow; set(src,'Enable','on');
        catch
        end
        fn();
    end

    % ---------------- loading ------------------------------------------------
    function uiLoadVideo()
        [fn,fp] = uigetfile( ...
            {'*.mp4;*.mov;*.avi;*.m4v;*.mkv;*.wmv','Video files';'*.*','All files'}, ...
            'Select a video file');
        if isequal(fn,0), return; end
        loadVideo(fullfile(fp,fn));
    end

    function loadVideo(path)
        try
            v = VideoReader(path);
        catch err
            errordlg(['Could not open video: ' err.message],'Load error'); return;
        end
        try
            [a,fsr] = audioread(path);
        catch
            errordlg(['Could not read audio from this file. On some systems ' ...
                'AUDIOREAD cannot decode compressed video audio; extract a .wav ' ...
                'and load that instead.'],'Audio error'); return;
        end
        stopPlay(); commitEditQuiet();
        vid = v; videoPath = path; fs = fsr;
        if size(a,2) > 1, a = mean(a,2); end
        audio = single(a(:)); audioMax = max(max(abs(audio)),eps);
        dur = numel(audio)/fs; maxFreq = min(5000, fs/2);
        events = struct('start',{},'end',{},'type',{},'transcript',{},'notes',{});
        currentEvent = []; selStart = NaN; selEnd = NaN; cursorTime = 0;
        viewStart = 0; viewEnd = min(dur,10);

        if ~isempty(placeholderTxt) && ishghandle(placeholderTxt)
            delete(placeholderTxt); placeholderTxt = [];
        end
        cla(axVideo);
        try
            vid.CurrentTime = 0; frame0 = readFrame(vid);
        catch
            frame0 = zeros(2,2,3,'uint8');
        end
        imgVideo = image(axVideo,frame0); axis(axVideo,'image','off');
        set(imgVideo,'PickableParts','none','HitTest','off');

        set(btnLoadAnnot,'Enable','on');
        [~,nm,ext] = fileparts(path);
        set(statusTxt,'String',sprintf('Loaded: %s%s   |   %.1f min, %d Hz audio, %.0f fps video', ...
            nm,ext,dur/60,fs,vid.FrameRate));
        refreshView(); showFrameAt(0);
    end

    % ---------------- view / drawing ----------------------------------------
    function refreshView()
        if isempty(vid), return; end
        viewStart = max(0,viewStart); viewEnd = min(dur,viewEnd);
        if viewEnd - viewStart < MINVIEW, viewEnd = min(dur,viewStart+MINVIEW); end
        set([axSpec axWave axTrans axNotes],'XLim',[viewStart viewEnd]);
        set([axSpec axWave axTrans],'XTick',[]); set(axNotes,'XTickMode','auto');
        safeDraw(@drawSpectrogram,'spectrogram');
        safeDraw(@drawWaveform,'waveform');
        safeDraw(@redrawEvents,'events');
        updateSelectionGraphics(); updateCursorGraphics();
    end

    function safeDraw(fn,label)
        try
            fn();
        catch err
            set(statusTxt,'String',['Draw error (' label '): ' err.message]);
        end
    end

    function drawSpectrogram()
        i0 = max(1,floor(viewStart*fs)+1); i1 = min(numel(audio),ceil(viewEnd*fs));
        if ~isempty(hSpecImg)&&ishghandle(hSpecImg), delete(hSpecImg); hSpecImg=[]; end
        if i1-i0 < 64, return; end
        seg = double(audio(i0:i1));
        winlen = max(64,round(0.006*fs)); if mod(winlen,2)==0, winlen=winlen+1; end
        Ncol = 700;
        hop  = max(1, floor((numel(seg)-winlen)/Ncol));
        nfft = 2^nextpow2(max(winlen,512));
        w    = 0.5 - 0.5*cos(2*pi*(0:winlen-1)'/(winlen-1));   % Hann window (no toolbox)
        st   = 1:hop:(numel(seg)-winlen+1);
        if isempty(st), return; end
        nf = floor(nfft/2)+1;
        P  = zeros(nf,numel(st));
        for c = 1:numel(st)
            idx = st(c):st(c)+winlen-1;
            X = fft(seg(idx).*w, nfft);
            P(:,c) = abs(X(1:nf));
        end
        P = 20*log10(P+eps);
        F = (0:nf-1)'*(fs/nfft);
        Tc = (st-1+winlen/2)/fs + viewStart;
        fmask = F <= maxFreq; if ~any(fmask), fmask = true(size(F)); end
        hSpecImg = imagesc(axSpec, Tc, F(fmask), P(fmask,:));
        set(axSpec,'YDir','normal','YLim',[0 maxFreq],'XLim',[viewStart viewEnd]);
        mx = max(P(fmask,:),[],'all');
        if isfinite(mx), setCLim(axSpec,[mx-70 mx]); end
        colormap(axSpec, flipud(gray(256)));
        set(hSpecImg,'PickableParts','none','HitTest','off'); uistack(hSpecImg,'bottom');
    end

    function setCLim(ax,lims)
        try, clim(ax,lims); catch, caxis(ax,lims); end %#ok<CAXIS>
    end

    function drawWaveform()
        i0 = max(1,floor(viewStart*fs)+1); i1 = min(numel(audio),ceil(viewEnd*fs));
        if ~isempty(hWave), delete(hWave(ishghandle(hWave))); hWave = []; end
        if i1 <= i0, return; end
        seg = double(audio(i0:i1)); tt = ((i0:i1)-1)/fs;
        if numel(seg) <= 4000
            hWave = plot(axWave,tt,seg/audioMax,'Color',[0 0 0]);
        else
            nb = 1500; edges = round(linspace(1,numel(seg)+1,nb+1));
            mins = zeros(1,nb); maxs = zeros(1,nb); tc = zeros(1,nb);
            for k = 1:nb
                idx = edges(k):edges(k+1)-1; if isempty(idx), idx = edges(k); end
                sk = seg(idx); mins(k)=min(sk); maxs(k)=max(sk); tc(k)=tt(idx(1));
            end
            xp = [tc fliplr(tc)]; yp = [maxs fliplr(mins)]/audioMax;
            hWave = patch(axWave,'XData',xp,'YData',yp,'FaceColor',[0.1 0.1 0.1], ...
                'EdgeColor',[0.1 0.1 0.1]);
        end
        set(hWave,'PickableParts','none','HitTest','off');
        set(axWave,'XLim',[viewStart viewEnd],'YLim',[-1 1]); uistack(hWave,'bottom');
    end

    function redrawEvents()
        delete(evGfx(ishghandle(evGfx))); evGfx = gobjects(0);
        vw = viewEnd - viewStart;
        for k = 1:numel(events)
            s = events(k).start; en = events(k).end;
            if en < viewStart || s > viewEnd, continue; end
            col = eventTypes(events(k).type).color;
            xs = [s en en s];
            yl = get(axSpec,'YLim');
            pSpec = patch(axSpec,'XData',xs,'YData',[yl(1) yl(1) yl(2) yl(2)], ...
                'FaceColor',col,'FaceAlpha',0.15,'EdgeColor',col,'LineWidth',0.5);
            pWave = patch(axWave,'XData',xs,'YData',[-1 -1 1 1], ...
                'FaceColor',col,'FaceAlpha',0.15,'EdgeColor',col,'LineWidth',0.5);
            pTr = patch(axTrans,'XData',xs,'YData',[0 0 1 1], ...
                'FaceColor',col,'FaceAlpha',0.20,'EdgeColor',col);
            pNo = patch(axNotes,'XData',xs,'YData',[0 0 1 1], ...
                'FaceColor',col,'FaceAlpha',0.20,'EdgeColor',col);
            tx0 = max(s,viewStart) + 0.004*vw;
            tTr = text(axTrans,tx0,0.5,events(k).transcript,'Clipping','on', ...
                'Interpreter','none','FontSize',8,'VerticalAlignment','middle');
            tNo = text(axNotes,tx0,0.5,events(k).notes,'Clipping','on', ...
                'Interpreter','none','FontSize',8,'VerticalAlignment','middle');
            if ~isempty(currentEvent) && currentEvent == k
                set([pSpec pWave],'LineWidth',1.75);
            end
            evGfx = [evGfx pSpec pWave pTr pNo tTr tNo]; %#ok<AGROW>
        end
        if ~isempty(evGfx), set(evGfx,'PickableParts','none','HitTest','off'); end
    end

    function updateSelectionGraphics()
        delete(selGfx(ishghandle(selGfx))); selGfx = gobjects(0);
        if isnan(selStart) || isnan(selEnd) || selEnd <= selStart, return; end
        for ax = [axSpec axWave]
            yl = get(ax,'YLim');
            p = patch(ax,'XData',[selStart selEnd selEnd selStart], ...
                'YData',[yl(1) yl(1) yl(2) yl(2)],'FaceColor',[0.4 0.4 0.4], ...
                'FaceAlpha',0.20,'EdgeColor',[0.3 0.3 0.3]);
            selGfx = [selGfx p]; %#ok<AGROW>
        end
        set(selGfx,'PickableParts','none','HitTest','off'); stackEach(selGfx,'top');
    end

    function updateCursorGraphics()
        delete(curGfx(ishghandle(curGfx))); curGfx = gobjects(0);
        if isnan(cursorTime), return; end
        l1 = plot(axSpec,[cursorTime cursorTime],get(axSpec,'YLim'),'--','Color',[0.9 0.5 0],'LineWidth',1);
        l2 = plot(axWave,[cursorTime cursorTime],[-1 1],'--','Color',[0.9 0.5 0],'LineWidth',1);
        curGfx = [l1 l2];
        set(curGfx,'PickableParts','none','HitTest','off'); stackEach(curGfx,'top');
    end

    % ---------------- zoom / scroll -----------------------------------------
    function setView(a,b)
        w = b - a;
        if w < MINVIEW, c=(a+b)/2; a=c-MINVIEW/2; b=c+MINVIEW/2; w=MINVIEW; end
        if w > dur, a=0; b=dur; end
        if a < 0,   b=b-a; a=0; end
        if b > dur, a=a-(b-dur); b=dur; end
        viewStart = max(0,a); viewEnd = min(dur,b); refreshView();
    end

    function zoomAbout(center,factor)
        if isnan(center), center = (viewStart+viewEnd)/2; end
        w = (viewEnd-viewStart)*factor; w = max(MINVIEW,min(dur,w));
        frac = (center-viewStart)/max(viewEnd-viewStart,eps); frac = max(0,min(1,frac));
        a = center - frac*w; setView(a,a+w);
    end

    function scrollBy(frac)
        sh = frac*(viewEnd-viewStart); setView(viewStart+sh,viewEnd+sh);
    end

    function c = cursorCenter()
        if isnan(cursorTime), c=(viewStart+viewEnd)/2; else, c=cursorTime; end
    end

    function t = pointerTime()
        try
            cp = get(axSpec,'CurrentPoint'); t = min(max(cp(1,1),viewStart),viewEnd);
        catch
            t = cursorCenter();
        end
    end

    % ---------------- mouse on spectrogram / waveform -----------------------
    function onAxDown(ax)
        if isempty(vid), return; end
        commitEdit();
        dragAxes = ax; cp = get(ax,'CurrentPoint');
        dragStartT = clampT(cp(1,1)); downPix = get(fig,'CurrentPoint'); didDrag = false;
        set(fig,'WindowButtonMotionFcn',@(~,~)cb(@onDrag),'WindowButtonUpFcn',@(~,~)cb(@onUp));
    end

    function onDrag()
        cp = get(dragAxes,'CurrentPoint'); t = clampT(cp(1,1));
        if norm(get(fig,'CurrentPoint')-downPix) > 4, didDrag = true; end
        if didDrag
            selStart = min(dragStartT,t); selEnd = max(dragStartT,t); currentEvent = [];
            updateSelectionGraphics();
        end
    end

    function onUp()
        set(fig,'WindowButtonMotionFcn','','WindowButtonUpFcn','');
        if didDrag
            cursorTime = selStart;
        else
            t = dragStartT; idx = eventAtTime(t);
            if ~isempty(idx)
                currentEvent = idx; selStart = events(idx).start; selEnd = events(idx).end;
                cursorTime = selStart;
            else
                selStart = NaN; selEnd = NaN; currentEvent = []; cursorTime = t;
            end
            redrawEvents();
        end
        updateSelectionGraphics(); updateCursorGraphics(); showFrameAt(cursorTime);
    end

    % ---------------- mouse on annotation panels ----------------------------
    function onPanelDown(ax,panel)
        if isempty(vid), return; end
        commitEdit();
        cp = get(ax,'CurrentPoint'); t = clampT(cp(1,1)); idx = eventAtTime(t);
        if ~isempty(idx)
            currentEvent = idx; selStart = events(idx).start; selEnd = events(idx).end;
            updateSelectionGraphics(); redrawEvents(); startEditing(idx,panel);
        end
    end

    % ---------------- keyboard ----------------------------------------------
    function onKey(e)
        if isempty(vid) || isEditing, return; end
        ctrl = ismember('control',e.Modifier); k = e.Key;
        if ctrl
            switch k
                case 'o', zoomAbout(cursorCenter(),2);
                case 'i', zoomAbout(cursorCenter(),0.5);
                case 'n'
                    if ~isnan(selStart)&&~isnan(selEnd)&&selEnd>selStart, setView(selStart,selEnd); end
                case 'a', setView(0,dur);
            end
            return;
        end
        switch k
            case 'space',  if isPlaying, pausePlay(); else, startPlay(); end
            case 'escape', stopPlay();
            otherwise
                for ti = 1:numel(eventTypes)
                    if strcmp(k,eventTypes(ti).key), addEvent(ti); break; end
                end
        end
    end

    function onScroll(e)
        if isempty(vid) || isEditing, return; end
        ctrl = ismember('control',get(fig,'CurrentModifier'));
        c = e.VerticalScrollCount;
        if ctrl, zoomAbout(pointerTime(),1.2^c); else, scrollBy(0.15*c); end
    end

    % ---------------- events -------------------------------------------------
    function addEvent(typeIdx)
        if isempty(vid) || isnan(selStart) || isnan(selEnd) || selEnd <= selStart, return; end
        s.start = selStart; s.end = selEnd; s.type = typeIdx; s.transcript = ''; s.notes = '';
        events(end+1) = s; newIdx = numel(events); currentEvent = newIdx;
        redrawEvents(); updateSelectionGraphics();
        startEditing(newIdx,'transcript');
    end

    function idx = eventAtTime(t)
        idx = [];
        for k = numel(events):-1:1
            if t >= events(k).start && t <= events(k).end, idx = k; return; end
        end
    end

    % ---------------- annotation text editing -------------------------------
    function startEditing(idx,panel)
        if isempty(vid), return; end
        commitEdit();
        editIdx = idx; editPanel = panel; isEditing = true;
        if strcmp(panel,'transcript')
            ax = axTrans; txt = events(idx).transcript; bg = [1 1 0.90];
        else
            ax = axNotes; txt = events(idx).notes;      bg = [0.90 1 1];
        end
        x0 = max(events(idx).start,viewStart); x1 = min(events(idx).end,viewEnd);
        set(editBox,'Position',dataRangeToPix(ax,x0,x1),'String',txt, ...
            'BackgroundColor',bg,'Visible','on','Enable','on');
        uicontrol(editBox);
    end

    function commitEdit()
        if ~isEditing, return; end
        isEditing = false;
        idx = editIdx;
        if isempty(idx), set(editBox,'Visible','off','Enable','off'); return; end
        str = get(editBox,'String');
        if iscell(str), str = strjoin(str,sprintf('\n'));
        elseif size(str,1) > 1, str = strjoin(cellstr(str),sprintf('\n')); end
        if strcmp(editPanel,'transcript'), events(idx).transcript = str;
        else,                              events(idx).notes      = str; end
        set(editBox,'Visible','off','Enable','off'); editIdx = []; redrawEvents();
    end

    function commitEditQuiet()
        if ~isEditing, return; end
        isEditing = false; set(editBox,'Visible','off','Enable','off'); editIdx = [];
    end

    function pos = dataRangeToPix(ax,x0,x1)
        axpix = getpixelposition(ax,true); xl = get(ax,'XLim');
        f0 = (x0-xl(1))/diff(xl); f1 = (x1-xl(1))/diff(xl);
        f0 = max(0,min(1,f0)); f1 = max(0,min(1,f1));
        px = axpix(1)+f0*axpix(3); pw = max(40,(f1-f0)*axpix(3));
        pos = [px axpix(2) pw axpix(4)];
    end

    % ---------------- playback ----------------------------------------------
    function startPlay()
        if isempty(vid) || isPlaying, return; end
        if ~isnan(selStart) && ~isnan(selEnd) && selEnd > selStart
            t0 = selStart; t1 = selEnd;
        elseif ~isnan(cursorTime)
            t0 = cursorTime; t1 = dur;
        else
            t0 = viewStart; t1 = dur;
        end
        s0 = max(1,round(t0*fs)+1); s1 = min(numel(audio),round(t1*fs));
        if s1 <= s0, return; end
        player = audioplayer(double(audio(s0:s1)),fs);
        playStartTime = (s0-1)/fs; playEndTime = t1; isPlaying = true;
        delete(playGfx(ishghandle(playGfx)));
        pl1 = plot(axSpec,[t0 t0],get(axSpec,'YLim'),'-','Color',[0.9 0 0.9],'LineWidth',1.5);
        pl2 = plot(axWave,[t0 t0],[-1 1],'-','Color',[0.9 0 0.9],'LineWidth',1.5);
        playGfx = [pl1 pl2]; set(playGfx,'PickableParts','none','HitTest','off');
        setappdata(fig,'player',player);
        % One seek to the start point; after this, frames are decoded forward.
        try
            vid.CurrentTime = max(0, min(t0, vid.Duration - 1/max(vid.FrameRate,1)));
        catch
        end
        play(player); start(playTimer);
    end

    function playTick()
        if ~isPlaying, return; end
        player = getappdata(fig,'player');
        if isempty(player) || ~isplaying(player), stopPlay(); return; end
        t = currentPlayTime(player);                     % audio = master clock
        if t >= playEndTime, stopPlay(); return; end
        if t > viewEnd || t < viewStart
            w = viewEnd-viewStart; viewStart = max(0,t-0.1*w); viewEnd = viewStart+w; refreshView();
        end
        if ~isempty(playGfx) && all(ishghandle(playGfx))
            set(playGfx(1),'XData',[t t]); set(playGfx(2),'XData',[t t]);
        end
        advanceFrameTo(t); drawnow limitrate;
    end

    function t = currentPlayTime(player)
        t = playStartTime + double(player.CurrentSample - 1)/fs;
    end

    function pausePlay()
        % Stop playback and leave the cursor where it stopped, so the next
        % space-bar press resumes from here (when no window is selected).
        if ~isPlaying, return; end
        player = getappdata(fig,'player');
        t = NaN;
        if ~isempty(player)
            try, t = currentPlayTime(player); catch, end
        end
        stopPlay();
        if isfinite(t) && (isnan(selStart) || isnan(selEnd) || selEnd <= selStart)
            cursorTime = clampT(t);
            updateCursorGraphics(); showFrameAt(cursorTime);
        end
    end

    function stopPlay()
        try, stop(playTimer); end %#ok<TRYNC>
        player = getappdata(fig,'player');
        if ~isempty(player), try, stop(player); end; end %#ok<TRYNC>
        isPlaying = false; delete(playGfx(ishghandle(playGfx))); playGfx = gobjects(0);
    end

    function advanceFrameTo(t)
        % Playback frame update: decode forward sequentially instead of seeking
        % every tick (seeking compressed video is slow and causes lag/drift).
        if isempty(vid) || isempty(imgVideo) || ~ishghandle(imgVideo), return; end
        fp = 1/max(vid.FrameRate,1);
        try
            % Only seek if we're behind, or too far ahead to decode through
            if t < vid.CurrentTime - fp || t > vid.CurrentTime + 0.5
                vid.CurrentTime = max(0, min(t, vid.Duration - fp));
            end
            fr = [];
            while hasFrame(vid) && vid.CurrentTime <= t
                fr = readFrame(vid);          % late frames are decoded but not drawn
            end
            if ~isempty(fr), set(imgVideo,'CData',fr); end
        catch
        end
    end

    function showFrameAt(t)
        % Random-access frame display (clicks, drags, pauses).
        if isempty(vid) || isempty(imgVideo) || ~ishghandle(imgVideo), return; end
        tt = max(0,min(t, vid.Duration - 1/max(vid.FrameRate,1)));
        try
            vid.CurrentTime = tt; set(imgVideo,'CData',readFrame(vid));
        catch
        end
    end

    % ---------------- save / load annotations -------------------------------
    function saveAnnotations()
        if isempty(vid), errordlg('Load a video first.','Save'); return; end
        commitEdit();
        [~,base] = fileparts(videoPath);
        defName = [base '_annot-disfluencies.xlsx'];
        [fn,fp] = uiputfile('*.xlsx','Save annotations',defName);
        if isequal(fn,0), return; end
        if isempty(events)
            starts = zeros(0,1); ends = zeros(0,1);
            types = cell(0,1); trans = cell(0,1); notesv = cell(0,1);
        else
            [~,ord] = sort([events.start]); ev = events(ord);
            starts = [ev.start]'; ends = [ev.end]';
            types  = arrayfun(@(x)eventTypes(x.type).name,ev,'UniformOutput',false)';
            trans  = arrayfun(@(x)x.transcript,ev,'UniformOutput',false)';
            notesv = arrayfun(@(x)x.notes,ev,'UniformOutput',false)';
        end
        T = table(starts,ends,types,trans,notesv, ...
            'VariableNames',{'starts','ends','event_type','transcript','notes'});
        try
            writetable(T,fullfile(fp,fn));
            set(statusTxt,'String',['Saved ' num2str(numel(events)) ' events to ' fn]);
        catch err
            errordlg(['Could not save: ' err.message],'Save error');
        end
    end

    function loadAnnotations()
        if isempty(vid), return; end
        [fn,fp] = uigetfile({'*.xlsx','Annotation table'},'Load annotations');
        if isequal(fn,0), return; end
        try
            T = readtable(fullfile(fp,fn));
        catch err
            errordlg(['Could not read file: ' err.message],'Load error'); return;
        end
        req = {'starts','ends','event_type','transcript','notes'};
        vn = lower(T.Properties.VariableNames); col = zeros(1,5);
        for j = 1:5, m = find(strcmp(vn,req{j}),1); if ~isempty(m), col(j)=m; end; end
        if any(col==0)
            if width(T) >= 5, col = 1:5;
            else, errordlg('File does not match the expected structure (need 5 columns).', ...
                    'Invalid file'); return; end
        end
        if ~isempty(events)
            ans_ = questdlg('annotations already present - replace current annotations with those in loaded file?', ...
                'Replace annotations?','OK','Cancel','Cancel');
            if ~strcmp(ans_,'OK'), return; end
        end
        commitEditQuiet();
        newEv = struct('start',{},'end',{},'type',{},'transcript',{},'notes',{});
        for i = 1:height(T)
            st = toNum(T{i,col(1)}); en = toNum(T{i,col(2)});
            nm = toStr(T{i,col(3)}); tr = toStr(T{i,col(4)}); nt = toStr(T{i,col(5)});
            if isnan(st) || isnan(en) || en <= st, continue; end
            ti = find(strcmpi({eventTypes.name},nm),1);
            if isempty(ti)
                eventTypes(end+1) = struct('name',nm,'color',[0.5 0.5 0.5],'key',''); %#ok<AGROW>
                ti = numel(eventTypes);
            end
            newEv(end+1) = struct('start',st,'end',en,'type',ti,'transcript',tr,'notes',nt); %#ok<AGROW>
        end
        events = newEv; currentEvent = []; selStart = NaN; selEnd = NaN;
        refreshView();
        set(statusTxt,'String',['Loaded ' num2str(numel(events)) ' events from ' fn]);
    end

    % ---------------- shortcuts window --------------------------------------
    function showShortcuts()
        evLines = '';
        for ti = 1:numel(eventTypes)
            if isempty(eventTypes(ti).key), keyd='(none)'; else, keyd=eventTypes(ti).key; end
            evLines = [evLines sprintf('   %-6s add "%s" event\n',keyd,eventTypes(ti).name)]; %#ok<AGROW>
        end
        msg = sprintf([ ...
            'ZOOM / SCROLL (spectrogram + waveform)\n' ...
            '   Ctrl+I / Ctrl+O    zoom in / out\n' ...
            '   Ctrl + wheel       zoom about pointer\n' ...
            '   wheel              scroll in time\n' ...
            '   Ctrl+N             zoom to selection\n' ...
            '   Ctrl+A             zoom to full recording\n\n' ...
            'SELECTION / CURSOR\n' ...
            '   click              place cursor (no selection)\n' ...
            '   click + drag       select a time window\n' ...
            '   click an event     make that window the current selection\n' ...
            '   drag over an event ignore it, make a fresh selection\n\n' ...
            'PLAYBACK\n' ...
            '   Space              play (window if selected, else from cursor)\n' ...
            '   Space again        pause (cursor stays where it stopped)\n' ...
            '   Esc                stop\n\n' ...
            'ANNOTATION\n' ...
            '   select a window, then press an event key below\n' ...
            '   type transcript; click Notes panel to type notes\n' ...
            '   click an event, then click a panel to edit its text\n' ...
            '   click the timeline to leave edit mode\n\n' ...
            'EVENT KEYS\n%s\n' ...
            'Event times come from the audio sample clock.'],evLines);
        h = msgbox(msg,'Shortcuts');
        try, set(findall(h,'Type','text'),'FontName','FixedWidth'); catch, end
    end

    % ---------------- helpers ------------------------------------------------
    function t = clampT(t), t = max(0,min(dur,t)); end

    % uistack requires all objects to share a parent. Our cursor/selection
    % handles span axSpec and axWave, so restack each one individually.
    function stackEach(objs,where)
        objs = objs(ishghandle(objs));
        for o = reshape(objs,1,[])
            uistack(o,where);
        end
    end

    function x = toNum(v)
        if iscell(v), v = v{1}; end
        if isnumeric(v), x = double(v); else, x = str2double(string(v)); end
    end

    function s = toStr(v)
        if iscell(v), v = v{1}; end
        if isnumeric(v)
            if isnan(v), s = ''; else, s = num2str(v); end
        else
            s = char(string(v)); if strcmpi(s,'NaN')||strcmp(s,'<missing>'), s=''; end
        end
    end

    function onClose()
        try, stop(playTimer); end %#ok<TRYNC>
        try, delete(playTimer); end %#ok<TRYNC>
        player = getappdata(fig,'player');
        if ~isempty(player), try, stop(player); end; end %#ok<TRYNC>
        delete(fig);
    end
end