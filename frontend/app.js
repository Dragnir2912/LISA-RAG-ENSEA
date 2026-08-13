const state = {
  conversations: [],
  currentId: null,
  messages: [],
  streaming: false,
  abortController: null,
  selectedFile: null,
  activeJobId: null,
  jobTimer: null,
  user: null,
  csrfToken: null,
  setupRequired: false,
  authMode: "login",
  authCapabilities: { registration_enabled: false, guest_enabled: false, password_reset_enabled: false },
  resetToken: new URLSearchParams(window.location.search).get("reset_token") || "",
  adminData: null,
  adminTimer: null,
  adminJobStatuses: new Map(),
  activeAdminJobId: null,
  activeOperationJobId: null,
  operationJobKind: null,
  facebookJobKind: "collect_facebook",
  facebookPagesSignature: "",
  healthTimer: null,
  language: localStorage.getItem("rag-ensea-language") || "fr",
  thinkingEnabled: localStorage.getItem("rag-ensea-thinking") === "true",
};

const translations = {
  fr: {
    newChat: "Nouvelle discussion", tools: "Outils", settings: "Paramètres", thisWeek: "Cette semaine",
    assistantTitle: "Assistant documentaire de l’ENSEA", messagePlaceholder: "Envoyer un message",
    toolsTitle: "Outils", toolsSubtitle: "Accédez aux outils de votre assistant ENSEA.", documents: "Documents",
    documentsDescription: "Ajoutez et recherchez des rapports, articles et publications de l’école.", openLibrary: "Ouvrir la bibliothèque",
    supervision: "Supervision", supervisionDescription: "Consultez les performances de recherche, de génération et les retours.", openDashboard: "Voir les performances",
    enseaIndex: "Index ENSEA", enseaIndexDescription: "Pages officielles, rapports et sources institutionnelles.", indexedPassages: "4 459 passages indexés",
    clubPublications: "Publications des clubs", clubDescription: "Publications des clubs et associations de l’ENSEA.", indexedPublications: "250 publications indexées",
    localAiModel: "Modèle d’IA local", localAiDescription: "Modèle d’IA exécuté sur cet ordinateur pour générer les réponses.", ready: "Prêt",
    brandSubtitle: "Assistant documentaire intelligent", language: "Langue", languageDescription: "Choisissez la langue de l’interface.",
    appearance: "Apparence", appearanceDescription: "Basculez entre les thèmes sombre et clair.", dark: "Sombre", light: "Clair",
    documentSync: "Synchronisation des documents", documentSyncDescription: "Les PDF ajoutés sont intégrés à l’index documentaire.", enabled: "Activée",
    localAccess: "Accès local", localAccessDescription: "Les conversations, modèles et documents restent sur cet ordinateur.", local: "Local",
    documentLocation: "Emplacement des documents", documentLocationDescription: "Dossier de stockage des documents PDF ajoutés.", browse: "Parcourir",
    library: "BIBLIOTHÈQUE", addDocuments: "Ajouter des documents", dropPdf: "Déposez un fichier PDF", dropPdfHint: "ou cliquez pour parcourir · 50 Mo maximum",
    title: "Titre", documentTitlePlaceholder: "Titre du document", authorDepartment: "Auteur ou service", authorPlaceholder: "Ex. Direction de la recherche",
    date: "Date", category: "Catégorie", report: "Rapport", researchArticle: "Article scientifique", dissertation: "Mémoire",
    administrative: "Administratif", training: "Formation", other: "Autre", addAndIndex: "Ajouter et indexer", processing: "Traitement",
    preparingDocument: "Préparation du document…", indexedDocuments: "Documents indexés", observability: "OBSERVABILITÉ", localSupervision: "Supervision locale",
    noSearchResult: "Aucun résultat.", conversationsHere: "Vos conversations apparaîtront ici.", delete: "Supprimer", waitForAnswer: "Attendez la fin de la réponse.",
    deleteConversation: "Supprimer « {title} » ?", thisConversation: "cette conversation", newTitle: "Nouveau titre", sourcesUsed: "Sources utilisées",
    useful: "Utile", improve: "À améliorer", copy: "Copier", you: "Vous", searchingSources: "Recherche des sources…", conversationReply: "Préparation de la réponse…", answerCopied: "Réponse copiée.",
    thinkingMode: "Réflexion", thinkingModeOn: "Mode réflexion activé : la réponse sera plus approfondie mais plus lente.", thinkingModeOff: "Mode réflexion désactivé.", deepReasoning: "Réflexion approfondie…",
    reasoningInProgress: "Réflexion en cours", reasoningComplete: "Réflexion terminée en {seconds} s", reasoningInterrupted: "Limite de réflexion atteinte · réponse finale sécurisée", showReasoning: "Afficher la réflexion complète", hideReasoning: "Masquer la réflexion complète",
    reasoningSummaryGeneral: "Analyse de la demande et préparation de la réponse", reasoningSummarySources: "Analyse des sources et vérification des références", reasoningSummaryChronology: "Reconstitution de la chronologie et vérification des dates", reasoningSummaryEnumeration: "Recensement des éléments et contrôle de la couverture", reasoningSummaryMembership: "Vérification des appartenances et des relations", reasoningSummaryCalculation: "Vérification des données et du calcul",
    feedbackSaved: "Merci, votre retour a été enregistré.", checkingIndex: "Consultation de l’index {route}…", displayStopped: "Affichage interrompu.", answerStopped: "Réponse interrompue.",
    pipelineError: "Erreur du pipeline", loading: "Chargement…", processingNow: "Traitement en cours", systemReady: "Système prêt", apiAvailable: "API disponible",
    localPipeline: "Pipeline local", apiUnavailable: "API indisponible", checkLocalServer: "Vérifiez le serveur local", loadingPipeline: "Chargement du pipeline…",
    selectPdf: "Sélectionnez un fichier PDF.", fileTooLarge: "Ce fichier dépasse la limite de 50 Mo.", documentReady: "Document prêt", importStopped: "Import interrompu",
    indexing: "Indexation en cours", processingDocument: "Traitement du document…", queued: "En attente", indexStatus: "Indexation", error: "Erreur", pagesPassages: "{pages} pages · {chunks} passages",
    noPdf: "Aucun PDF ajouté", importedHere: "Les documents importés apparaîtront ici.", sendingPdf: "Envoi du PDF…", pdfIndexed: "Le PDF est indexé et peut maintenant être interrogé.", indexingFailed: "L’indexation a échoué.",
    conversations: "Conversations", questions: "Questions", pdfDocuments: "Documents PDF", addedPassages: "Passages ajoutés", averageTime: "Temps moyen", satisfaction: "Satisfaction",
    sourceBreakdown: "Répartition des sources", noRecordedAnswer: "Aucune réponse enregistrée pour le moment.", firstToken: "Premier mot", throughput: "Débit moyen", activity: "ACTIVITÉ", globalObservability: "OBSERVABILITÉ GLOBALE",
    toolsSubtitleUser: "Retrouvez les documents, votre activité et la sécurité de votre compte.", toolsSubtitleAdmin: "Accédez au travail documentaire et au pilotage complet du système.",
    userRole: "Utilisateur", adminRole: "Administrateur", workspaceTools: "VOTRE ESPACE", workspaceToolsDescription: "Les fonctions utiles pendant vos recherches.",
    documentLibrary: "Bibliothèque documentaire", documentsUserDescription: "Consultez les rapports, articles et documents partagés.", documentsAdminDescription: "Ajoutez, contrôlez et consultez les documents indexés.",
    browseLibrary: "Parcourir", manageLibrary: "Gérer", myActivity: "Mon activité", globalSupervision: "Supervision globale", activityUserDescription: "Consultez vos conversations, retours et temps de réponse.", activityAdminDescription: "Suivez l’usage, les performances et les retours du système.", view: "Consulter",
    sessionsSecurity: "Sessions et sécurité", sessionsSecurityDescription: "Contrôlez les appareils connectés à votre compte.", manage: "Gérer",
    systemAdministration: "ADMINISTRATION DU SYSTÈME", systemAdministrationDescription: "Collecte, index, accès et configuration du RAG.",
    sourcesAndIndex: "Sources et index", sourcesAndIndexDescription: "Collecter, nettoyer, chunker et indexer", scheduledJobs: "Planifications", scheduledJobsDescription: "Automatiser les mises à jour documentaires",
    modelsAndPerformance: "Modèles et performances", modelsAndPerformanceDescription: "Choisir le modèle et le profil matériel", usersAndAccess: "Utilisateurs et accès", usersAndAccessDescription: "Créer des comptes et attribuer les rôles",
    availableKnowledge: "CONNAISSANCES DISPONIBLES", availableKnowledgeDescription: "État réel des corpus consultés par l’assistant.", institutionalSources: "Sources ENSEA", institutionalSourcesDescription: "Pages officielles et PDF institutionnels",
    economicSources: "Sources économiques", economicSourcesDescription: "Rapports ANSTAT et bibliothèque ECO RCI", socialSources: "Vie associative", socialSourcesDescription: "Publications des clubs et associations", answerModel: "Modèle de réponse",
    passages: "passages", unavailable: "Indisponible", connectingSystem: "Connexion au système…", systemOperational: "Système opérationnel", maintenanceRunning: "Maintenance en cours", systemError: "Pipeline indisponible", cpuProfile: "Profil CPU uniquement", optimizedProfile: "Profil RTX 8 Go optimisé", sharedGpuProfile: "Profil GPU partagé",
    interfacePreferences: "INTERFACE", interfacePreferencesDescription: "Préférences propres à votre compte sur cet appareil.", accessAndPrivacy: "ACCÈS ET CONFIDENTIALITÉ", accessAndPrivacyDescription: "Droits du compte, documents et sessions connectées.",
    accountAccess: "Type de compte", userAccessDescription: "Accès aux conversations privées et à la bibliothèque partagée.", adminAccessDescription: "Accès complet aux conversations, utilisateurs et opérations du système.",
    privateConversations: "Conversations privées", privateConversationsDescription: "Vous seul pouvez consulter l’historique associé à ce compte.", protected: "Protégées", userDocumentsSetting: "Consultation des documents partagés par les administrateurs.", adminDocumentsSetting: "Ajout, indexation et consultation des documents du système.", open: "Ouvrir",
    connectedSessions: "Sessions connectées", connectedSessionsDescription: "Consultez et révoquez les autres connexions à votre compte.", localData: "Données locales", localDataDescription: "Les conversations, documents et modèles restent sur le serveur ENSEA.",
    administrationSettings: "ADMINISTRATION", administrationSettingsDescription: "Les réglages qui affectent tous les utilisateurs.", configure: "Configurer", collectionAndSchedules: "Collecte et planifications", collectionAndSchedulesDescription: "Pilotez les sources et les mises à jour automatiques."
  },
  en: {
    newChat: "New Chat", tools: "Launch", settings: "Settings", thisWeek: "This week",
    assistantTitle: "ENSEA document assistant", messagePlaceholder: "Send a message",
    toolsTitle: "Launch", toolsSubtitle: "Open a tool for your ENSEA assistant.", documents: "Documents",
    documentsDescription: "Add and search reports, articles and school publications.", openLibrary: "Open document library",
    supervision: "Supervision", supervisionDescription: "Inspect retrieval, generation and feedback performance.", openDashboard: "View performance dashboard",
    enseaIndex: "ENSEA Index", enseaIndexDescription: "Official pages, reports and institutional sources.", indexedPassages: "4,459 indexed passages",
    clubPublications: "Club publications", clubDescription: "Posts from ENSEA clubs and associations.", indexedPublications: "250 indexed publications",
    localAiModel: "Local AI model", localAiDescription: "AI model running on this computer to generate answers.", ready: "Ready",
    brandSubtitle: "Intelligent document assistant", language: "Language", languageDescription: "Choose the interface language.",
    appearance: "Appearance", appearanceDescription: "Switch between dark and light themes.", dark: "Dark", light: "Light",
    documentSync: "Document synchronization", documentSyncDescription: "Uploaded PDFs are integrated into the document index.", enabled: "Enabled",
    localAccess: "Local access", localAccessDescription: "Conversations, models and documents remain on this computer.", local: "Local",
    documentLocation: "Document location", documentLocationDescription: "Folder where uploaded PDF documents are stored.", browse: "Browse",
    library: "LIBRARY", addDocuments: "Add documents", dropPdf: "Drop a PDF file", dropPdfHint: "or click to browse · 50 MB maximum",
    title: "Title", documentTitlePlaceholder: "Document title", authorDepartment: "Author or department", authorPlaceholder: "e.g. Research department",
    date: "Date", category: "Category", report: "Report", researchArticle: "Research article", dissertation: "Dissertation",
    administrative: "Administrative", training: "Training", other: "Other", addAndIndex: "Add and index", processing: "Processing",
    preparingDocument: "Preparing document…", indexedDocuments: "Indexed documents", observability: "OBSERVABILITY", localSupervision: "Local supervision",
    noSearchResult: "No results.", conversationsHere: "Your conversations will appear here.", delete: "Delete", waitForAnswer: "Wait for the answer to finish.",
    deleteConversation: "Delete “{title}”?", thisConversation: "this conversation", newTitle: "New title", sourcesUsed: "Sources used",
    useful: "Helpful", improve: "Needs improvement", copy: "Copy", you: "You", searchingSources: "Searching sources…", conversationReply: "Preparing the reply…", answerCopied: "Answer copied.",
    thinkingMode: "Reasoning", thinkingModeOn: "Reasoning mode enabled: the answer will be more thorough but slower.", thinkingModeOff: "Reasoning mode disabled.", deepReasoning: "Reasoning more deeply…",
    reasoningInProgress: "Reasoning in progress", reasoningComplete: "Reasoning completed in {seconds} s", reasoningInterrupted: "Reasoning limit reached · final answer recovered", showReasoning: "Show full reasoning", hideReasoning: "Hide full reasoning",
    reasoningSummaryGeneral: "Analyzing the request and preparing the answer", reasoningSummarySources: "Analyzing sources and checking references", reasoningSummaryChronology: "Reconstructing the timeline and checking dates", reasoningSummaryEnumeration: "Identifying items and checking coverage", reasoningSummaryMembership: "Checking memberships and relationships", reasoningSummaryCalculation: "Checking the data and calculation",
    feedbackSaved: "Thank you, your feedback was saved.", checkingIndex: "Searching the {route} index…", displayStopped: "Display stopped.", answerStopped: "Answer stopped.",
    pipelineError: "Pipeline error", loading: "Loading…", processingNow: "Processing", systemReady: "System ready", apiAvailable: "API available",
    localPipeline: "Local pipeline", apiUnavailable: "API unavailable", checkLocalServer: "Check the local server", loadingPipeline: "Loading pipeline…",
    selectPdf: "Select a PDF file.", fileTooLarge: "This file exceeds the 50 MB limit.", documentReady: "Document ready", importStopped: "Import stopped",
    indexing: "Indexing", processingDocument: "Processing document…", queued: "Queued", indexStatus: "Indexing", error: "Error", pagesPassages: "{pages} pages · {chunks} passages",
    noPdf: "No PDF added", importedHere: "Imported documents will appear here.", sendingPdf: "Uploading PDF…", pdfIndexed: "The PDF is indexed and can now be queried.", indexingFailed: "Indexing failed.",
    conversations: "Conversations", questions: "Questions", pdfDocuments: "PDF documents", addedPassages: "Added passages", averageTime: "Average time", satisfaction: "Satisfaction",
    sourceBreakdown: "Source breakdown", noRecordedAnswer: "No answer has been recorded yet.", firstToken: "First token", throughput: "Average throughput", activity: "ACTIVITY", globalObservability: "GLOBAL OBSERVABILITY",
    toolsSubtitleUser: "Access documents, your activity and account security.", toolsSubtitleAdmin: "Access document work and full system administration.",
    userRole: "User", adminRole: "Administrator", workspaceTools: "YOUR WORKSPACE", workspaceToolsDescription: "Useful features for your research.",
    documentLibrary: "Document library", documentsUserDescription: "Browse shared reports, articles and documents.", documentsAdminDescription: "Add, review and browse indexed documents.",
    browseLibrary: "Browse", manageLibrary: "Manage", myActivity: "My activity", globalSupervision: "Global supervision", activityUserDescription: "Review your conversations, feedback and response times.", activityAdminDescription: "Monitor usage, performance and system feedback.", view: "View",
    sessionsSecurity: "Sessions and security", sessionsSecurityDescription: "Control the devices connected to your account.", manage: "Manage",
    systemAdministration: "SYSTEM ADMINISTRATION", systemAdministrationDescription: "Collection, indexes, access and RAG configuration.",
    sourcesAndIndex: "Sources and indexes", sourcesAndIndexDescription: "Collect, clean, chunk and index", scheduledJobs: "Schedules", scheduledJobsDescription: "Automate document updates",
    modelsAndPerformance: "Models and performance", modelsAndPerformanceDescription: "Choose the model and hardware profile", usersAndAccess: "Users and access", usersAndAccessDescription: "Create accounts and assign roles",
    availableKnowledge: "AVAILABLE KNOWLEDGE", availableKnowledgeDescription: "Live status of the corpora searched by the assistant.", institutionalSources: "ENSEA sources", institutionalSourcesDescription: "Official pages and institutional PDFs",
    economicSources: "Economic sources", economicSourcesDescription: "ANSTAT reports and ECO RCI library", socialSources: "Community life", socialSourcesDescription: "Club and association publications", answerModel: "Answer model",
    passages: "passages", unavailable: "Unavailable", connectingSystem: "Connecting to the system…", systemOperational: "System operational", maintenanceRunning: "Maintenance running", systemError: "Pipeline unavailable", cpuProfile: "CPU-only profile", optimizedProfile: "Optimized 8 GB RTX profile", sharedGpuProfile: "Shared GPU profile",
    interfacePreferences: "INTERFACE", interfacePreferencesDescription: "Account preferences on this device.", accessAndPrivacy: "ACCESS AND PRIVACY", accessAndPrivacyDescription: "Account rights, documents and connected sessions.",
    accountAccess: "Account type", userAccessDescription: "Access to private conversations and the shared library.", adminAccessDescription: "Full access to conversations, users and system operations.",
    privateConversations: "Private conversations", privateConversationsDescription: "Only you can access the history linked to this account.", protected: "Protected", userDocumentsSetting: "Browse documents shared by administrators.", adminDocumentsSetting: "Add, index and browse system documents.", open: "Open",
    connectedSessions: "Connected sessions", connectedSessionsDescription: "Review and revoke other connections to your account.", localData: "Local data", localDataDescription: "Conversations, documents and models remain on the ENSEA server.",
    administrationSettings: "ADMINISTRATION", administrationSettingsDescription: "Settings that affect every user.", configure: "Configure", collectionAndSchedules: "Collection and schedules", collectionAndSchedulesDescription: "Manage sources and automated updates."
  }
};

function t(key, variables = {}) {
  const template = translations[state.language]?.[key] || translations.fr[key] || key;
  return Object.entries(variables).reduce((value, [name, replacement]) => value.replaceAll(`{${name}}`, replacement), template);
}

const $ = (selector) => document.querySelector(selector);
const elements = {
  authScreen: $("#auth-screen"), authForm: $("#auth-form"), authTitle: $("#auth-title"),
  authDescription: $("#auth-description"), authNameField: $("#auth-name-field"), authName: $("#auth-name"),
  authEmailField: $("#auth-email-field"), authPasswordField: $("#auth-password-field"), authConfirmField: $("#auth-confirm-field"),
  authEmail: $("#auth-email"), authPassword: $("#auth-password"), authConfirm: $("#auth-confirm"), authError: $("#auth-error"), authSubmit: $("#auth-submit"),
  authLinks: $("#auth-links"), authForgot: $("#auth-forgot"), authRegister: $("#auth-register"), authGuest: $("#auth-guest"), authBack: $("#auth-back"),
  shell: $("#app-shell"),
  main: $(".main-panel"),
  sidebar: $("#sidebar"), sidebarBackdrop: $("#sidebar-backdrop"), sidebarClose: $("#sidebar-close"),
  menuButton: $("#menu-button"), collapsedNewChat: $("#collapsed-new-chat"), newChat: $("#new-chat-button"), launchButton: $("#launch-button"), search: $("#conversation-search"),
  conversationList: $("#conversation-list"), conversationTitle: $("#conversation-title"), renameButton: $("#rename-button"),
  welcome: $("#welcome-state"), welcomeTitle: $("#welcome-title"), welcomeSubtitle: $("#welcome-subtitle"), messages: $("#messages"), chatScroll: $("#chat-scroll"),
  chatView: $("#chat-view"), launchView: $("#launch-view"),
  toolsSubtitle: $("#tools-subtitle"), toolsRoleBadge: $("#tools-role-badge"),
  toolDocumentsTitle: $("#tool-documents-title"), toolDocumentsDescription: $("#tool-documents-description"), toolDocumentsAction: $("#tool-documents-action"),
  toolMetricsTitle: $("#tool-metrics-title"), toolMetricsDescription: $("#tool-metrics-description"), toolMetricsAction: $("#tool-metrics-action"), toolSessionsButton: $("#tool-sessions-button"),
  toolInstitutionalCount: $("#tool-institutional-count"), toolEconomicCount: $("#tool-economic-count"), toolFacebookCount: $("#tool-facebook-count"), toolModelName: $("#tool-model-name"), toolModelProfile: $("#tool-model-profile"), toolPipelineDot: $("#tool-pipeline-dot"), toolPipelineStatus: $("#tool-pipeline-status"),
  composer: $("#composer-form"), input: $("#message-input"), sendButton: $("#send-button"), sendIcon: $("#send-icon"), thinkingToggle: $("#thinking-toggle"),
  statusDot: $("#status-dot"), statusLabel: $("#status-label"), statusDetail: $("#status-detail"), modelName: $("#model-name"),
  documentsButton: $("#documents-button"), documentsDialog: $("#documents-dialog"), documentsClose: $("#documents-close"), documentsDialogTitle: $("#documents-dialog-title"),
  attachButton: $("#attach-button"), dropZone: $("#drop-zone"), pdfInput: $("#pdf-input"), selectedFile: $("#selected-file"),
  documentTitle: $("#document-title"), documentAuthor: $("#document-author"), documentDate: $("#document-date"),
  documentCategory: $("#document-category"), uploadButton: $("#upload-button"), uploadProgress: $("#upload-progress"),
  progressTitle: $("#progress-title"), progressValue: $("#progress-value"), progressFill: $("#progress-fill"),
  progressMessage: $("#progress-message"), documentList: $("#document-list"), documentCount: $("#document-count"),
  metricsButton: $("#metrics-button"), metricsDialog: $("#metrics-dialog"), metricsClose: $("#metrics-close"), metricsDialogKicker: $("#metrics-dialog-kicker"), metricsDialogTitle: $("#metrics-dialog-title"),
  metricGrid: $("#metric-grid"), routeBreakdown: $("#route-breakdown"),
  settingsButton: $("#settings-button"), settingsDialog: $("#settings-dialog"), settingsClose: $("#settings-close"),
  settingsDocumentsButton: $("#settings-documents-button"), settingsRoleBadge: $("#settings-role-badge"), settingsAccessRole: $("#settings-access-role"), settingsAccessDescription: $("#settings-access-description"), settingsDocumentsDescription: $("#settings-documents-description"),
  accountButton: $("#account-button"), accountAvatar: $("#account-avatar"), accountName: $("#account-name"), accountRole: $("#account-role"),
  settingsAccountName: $("#settings-account-name"), settingsAccountEmail: $("#settings-account-email"), logoutButton: $("#logout-button"),
  sessionsButton: $("#sessions-button"), sessionsDialog: $("#sessions-dialog"), sessionsClose: $("#sessions-close"), sessionsList: $("#sessions-list"),
  adminButton: $("#admin-button"), adminDialog: $("#admin-dialog"), adminClose: $("#admin-close"), adminHealth: $("#admin-health"),
  adminRefresh: $("#admin-refresh"), adminMetrics: $("#admin-metrics"), operationGrid: $("#operation-grid"),
  operationConfigurator: $("#operation-configurator"), operationKicker: $("#operation-kicker"), operationTitle: $("#operation-title"), operationDescription: $("#operation-description"), operationClose: $("#operation-close"), operationForm: $("#operation-form"), operationParameters: $("#operation-parameters"), operationImpact: $("#operation-impact"), operationCancel: $("#operation-cancel"), operationLaunch: $("#operation-launch"),
  operationJobStatus: $("#operation-job-status"), operationJobTitle: $("#operation-job-title"), operationJobMessage: $("#operation-job-message"), operationJobPercent: $("#operation-job-percent"), operationJobFill: $("#operation-job-fill"), operationJobDetail: $("#operation-job-detail"),
  facebookCollector: $("#facebook-collector"), facebookCollectorTitle: $("#facebook-collector-title"), facebookCollectorDescription: $("#facebook-collector-description"), facebookCollectorClose: $("#facebook-collector-close"), facebookCollectionForm: $("#facebook-collection-form"),
  facebookTokenState: $("#facebook-token-state"), facebookTokenField: $("#facebook-token-field"), facebookToken: $("#facebook-token"), facebookTokenHint: $("#facebook-token-hint"), facebookRememberRow: $("#facebook-remember-row"), facebookRememberToken: $("#facebook-remember-token"),
  facebookPageOptions: $("#facebook-page-options"), facebookSelectionSummary: $("#facebook-selection-summary"), facebookSelectAll: $("#facebook-select-all"), facebookSelectNone: $("#facebook-select-none"),
  facebookMaxPosts: $("#facebook-max-posts"), facebookEstimate: $("#facebook-estimate"), facebookCancel: $("#facebook-cancel"), facebookLaunch: $("#facebook-launch"),
  facebookJobStatus: $("#facebook-job-status"), facebookJobTitle: $("#facebook-job-title"), facebookJobMessage: $("#facebook-job-message"), facebookJobPercent: $("#facebook-job-percent"), facebookJobFill: $("#facebook-job-fill"), facebookJobDetail: $("#facebook-job-detail"),
  scheduleForm: $("#schedule-form"), scheduleName: $("#schedule-name"), scheduleKind: $("#schedule-kind"), scheduleInterval: $("#schedule-interval"), scheduleList: $("#schedule-list"), scheduleFacebookConfig: $("#schedule-facebook-config"), scheduleFacebookPages: $("#schedule-facebook-pages"), scheduleFacebookMaxPosts: $("#schedule-facebook-max-posts"),
  modelForm: $("#model-form"), generationModel: $("#generation-model"), performanceProfile: $("#performance-profile"), modelInventory: $("#model-inventory"),
  userForm: $("#user-form"), newUserName: $("#new-user-name"), newUserEmail: $("#new-user-email"), newUserPassword: $("#new-user-password"), newUserRole: $("#new-user-role"), userList: $("#user-list"),
  securityMetrics: $("#security-metrics"), securitySessionList: $("#security-session-list"), securityAuditList: $("#security-audit-list"),
  jobList: $("#job-list"),
  toastRegion: $("#toast-region"),
};

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function renderMarkdown(value, withCitations = true) {
  let safe = escapeHtml(value).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  if (withCitations) safe = safe.replace(/\[(\d+)\]/g, (_, index) => `<button class="citation-ref" type="button" data-citation-index="${index}">[${index}]</button>`);
  return safe.split(/\n{2,}/).map((paragraph) => `<p>${paragraph.replaceAll("\n", "<br>")}</p>`).join("");
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  if (state.csrfToken && !["GET", "HEAD", "OPTIONS"].includes((options.method || "GET").toUpperCase())) headers["X-CSRF-Token"] = state.csrfToken;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = `Erreur ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch (_error) { /* réponse non JSON */ }
    if (response.status === 401 && state.user) showAuth(false);
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(message) {
  const node = document.createElement("div");
  node.className = "toast";
  node.textContent = message;
  elements.toastRegion.append(node);
  window.setTimeout(() => node.remove(), 4200);
}

function relativeDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const seconds = Math.round((date.getTime() - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat(state.language, { numeric: "auto" });
  const ranges = [[60, "second"], [60, "minute"], [24, "hour"], [7, "day"], [4.345, "week"], [12, "month"], [Infinity, "year"]];
  let duration = seconds;
  for (const [amount, unit] of ranges) {
    if (Math.abs(duration) < amount) return formatter.format(Math.round(duration), unit);
    duration /= amount;
  }
  return date.toLocaleDateString(state.language === "fr" ? "fr-FR" : "en-US");
}

function formatBytes(bytes) {
  if (!Number(bytes)) return state.language === "fr" ? "0 octet" : "0 B";
  const units = state.language === "fr" ? ["octets", "Ko", "Mo", "Go"] : ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / (1024 ** index);
  return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
}

function openSidebar() {
  if (window.innerWidth <= 960) {
    elements.sidebar.classList.add("open"); elements.sidebarBackdrop.classList.add("visible");
  } else {
    elements.shell.classList.remove("sidebar-collapsed");
  }
}
function closeSidebar() {
  if (window.innerWidth <= 960) {
    elements.sidebar.classList.remove("open"); elements.sidebarBackdrop.classList.remove("visible");
  }
}
function collapseSidebar() {
  if (window.innerWidth <= 760) closeSidebar();
  else elements.shell.classList.add("sidebar-collapsed");
}
function displayConversationTitle(value) {
  return ["Nouvelle discussion", "Nouvelle conversation", "New Chat"].includes(value) ? t("newChat") : value;
}
function showView(view) {
  const launch = view === "launch";
  elements.chatView.hidden = launch;
  elements.launchView.hidden = !launch;
  elements.newChat.classList.toggle("active", !launch);
  elements.launchButton.classList.toggle("active", launch);
  if (launch && state.user) updateHealth();
  closeSidebar();
}
function currentConversation() { return state.conversations.find((item) => item.id === state.currentId); }

function renderConversations() {
  const query = elements.search.value.trim().toLocaleLowerCase(state.language);
  const conversations = state.conversations.filter((item) => !query || item.title.toLocaleLowerCase(state.language).includes(query));
  elements.conversationList.innerHTML = "";
  if (!conversations.length) {
    elements.conversationList.innerHTML = `<div class="empty-conversations">${query ? t("noSearchResult") : t("conversationsHere")}</div>`;
    return;
  }
  for (const conversation of conversations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `conversation-item${conversation.id === state.currentId ? " active" : ""}`;
    const displayedTitle = displayConversationTitle(conversation.title);
    button.title = `${displayedTitle} · ${relativeDate(conversation.updated_at)}`;
    button.innerHTML = `<span class="conversation-item-title">${escapeHtml(displayedTitle)}</span><span class="conversation-delete" role="button" aria-label="${t("delete")}">×</span>`;
    button.addEventListener("click", (event) => {
      if (event.target.closest(".conversation-delete")) deleteConversation(conversation.id);
      else selectConversation(conversation.id);
    });
    elements.conversationList.append(button);
  }
}

async function loadConversations({ selectFirst = true } = {}) {
  state.conversations = await api("/api/conversations");
  renderConversations();
  if (selectFirst && !state.currentId && state.conversations.length) await selectConversation(state.conversations[0].id);
}

async function createConversation() {
  if (state.streaming) return toast(t("waitForAnswer"));
  const conversation = await api("/api/conversations", { method: "POST", body: JSON.stringify({ title: "Nouvelle conversation" }) });
  state.conversations.unshift(conversation);
  state.currentId = conversation.id;
  state.messages = [];
  renderConversations(); renderMessages(); showView("chat"); elements.input.focus();
}

function startNewChat() {
  if (state.streaming) return toast(t("waitForAnswer"));
  state.currentId = null;
  state.messages = [];
  renderConversations();
  renderMessages();
  showView("chat");
  elements.input.focus();
}

async function selectConversation(id) {
  if (state.streaming) return toast(t("waitForAnswer"));
  const conversation = await api(`/api/conversations/${id}`);
  state.currentId = conversation.id;
  state.messages = conversation.messages || [];
  const known = state.conversations.find((item) => item.id === conversation.id);
  if (known) Object.assign(known, conversation);
  renderConversations(); renderMessages(); showView("chat");
}

async function deleteConversation(id) {
  if (state.streaming) return;
  const item = state.conversations.find((conversation) => conversation.id === id);
  if (!window.confirm(t("deleteConversation", { title: item?.title || t("thisConversation") }))) return;
  await api(`/api/conversations/${id}`, { method: "DELETE" });
  state.conversations = state.conversations.filter((conversation) => conversation.id !== id);
  if (state.currentId === id) {
    state.currentId = null; state.messages = [];
    if (state.conversations.length) await selectConversation(state.conversations[0].id);
    else startNewChat();
  }
  renderConversations();
}

async function renameConversation() {
  const conversation = currentConversation();
  if (!conversation || state.streaming) return;
  const title = window.prompt(t("newTitle"), conversation.title)?.trim();
  if (!title || title === conversation.title) return;
  Object.assign(conversation, await api(`/api/conversations/${conversation.id}`, { method: "PATCH", body: JSON.stringify({ title }) }));
  renderConversations(); renderMessages();
}

function citationsMarkup(citations, messageId) {
  if (!citations?.length) return "";
  return `<div class="citations-block"><div class="citations-heading"><span>${t("sourcesUsed")}</span><span>${citations.length}</span></div>${citations.map((citation) => `
    <a class="citation-card" id="citation-${messageId}-${citation.index}" href="${escapeHtml(citation.url)}" target="_blank" rel="noopener noreferrer">
      <span class="citation-number">${citation.index}</span><span class="citation-copy"><strong>${escapeHtml(citation.title || "Source ENSEA")}</strong><span>${escapeHtml(citation.passage || citation.section || citation.url)}</span></span><span class="citation-open">↗</span>
    </a>`).join("")}</div>`;
}

const reasoningSummaryKeys = {
  general: "reasoningSummaryGeneral",
  sources: "reasoningSummarySources",
  chronology: "reasoningSummaryChronology",
  enumeration: "reasoningSummaryEnumeration",
  membership: "reasoningSummaryMembership",
  calculation: "reasoningSummaryCalculation",
};

function inferReasoningSummaryKey(content) {
  const normalized = (content || "").normalize("NFKC").toLowerCase();
  const categories = [
    ["calculation", /\b(calcul|calculate|computed?|taux|rate|variation|pourcentage|percentage|formula|formule)\b/],
    ["chronology", /\b(chronolog|succession|timeline|date|dates|période|periode|year|année|annee)\b/],
    ["enumeration", /\b(exhausti|liste|list|count|nombre|membres?|members?|finalistes?|clubs?|associations?)\b/],
    ["membership", /\b(appartient|membership|affiliation|associé|associe|associated|partenaire|partner)\b/],
    ["sources", /\b(sources?|citations?|références?|references?|passages?|documents?|evidence|preuves?)\b/],
  ];
  return categories.find(([, pattern]) => pattern.test(normalized))?.[0] || "general";
}

function reasoningSummary(message) {
  const key = message.reasoning?.summary_key || "general";
  return t(reasoningSummaryKeys[key] || reasoningSummaryKeys.general);
}

function reasoningMarkup(message) {
  const reasoning = message.reasoning;
  const active = Boolean(message.pending && reasoning?.active);
  if (!reasoning && !(message.pending && message.thinking)) return "";
  const content = reasoning?.content || "";
  const hasContent = Boolean(content.trim());
  const seconds = Number(reasoning?.seconds || 0).toFixed(1);
  const stateLabel = active
    ? t("reasoningInProgress")
    : reasoning?.truncated
      ? t("reasoningInterrupted")
      : t("reasoningComplete", { seconds });
  const actionLabel = hasContent ? t("showReasoning") : stateLabel;
  return `<section class="reasoning-trace ${active ? "active" : ""} ${hasContent ? "has-content" : ""}">
    <button class="reasoning-toggle" type="button" aria-expanded="false" aria-label="${escapeHtml(actionLabel)}" title="${escapeHtml(actionLabel)}" ${hasContent ? "" : "disabled"}>
      <span class="reasoning-mark" aria-hidden="true">✦</span>
      <span class="reasoning-copy"><span class="reasoning-state">${escapeHtml(stateLabel)}</span><span class="reasoning-summary ${active ? "shimmer" : ""}">${escapeHtml(reasoningSummary(message))}</span></span>
      <svg class="reasoning-chevron" aria-hidden="true" viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"></path></svg>
    </button>
    <div class="reasoning-details"><div class="reasoning-details-inner"><div class="reasoning-transcript">${escapeHtml(content)}</div></div></div>
  </section>`;
}

function metricsMarkup(message) {
  if (message.role !== "assistant" || message.pending) return "";
  const metrics = message.metrics || {};
  const feedback = message.feedback?.value;
  return `<div class="message-meta">
    ${message.route ? `<span class="meta-chip route">◎ ${escapeHtml(message.route)}</span>` : ""}
    ${metrics.thinking_enabled ? `<span class="meta-chip">✦ ${t("thinkingMode")}</span>` : ""}
    ${metrics.model_name ? `<span class="meta-chip">AI ${escapeHtml(metrics.model_name)}</span>` : ""}
    <span class="meta-chip">◷ ${Number(metrics.total_seconds || 0).toFixed(1)} s</span>
    ${Number(metrics.tokens_per_second || 0) ? `<span class="meta-chip">⚡ ${Number(metrics.tokens_per_second).toFixed(1)} tok/s</span>` : ""}
    <div class="message-actions"><button class="message-action feedback-up ${feedback === "up" ? "active" : ""}" data-message-id="${message.id}">${t("useful")}</button><button class="message-action down feedback-down ${feedback === "down" ? "active" : ""}" data-message-id="${message.id}">${t("improve")}</button><button class="message-action copy-answer" data-message-id="${message.id}">${t("copy")}</button></div>
  </div>`;
}

function messageMarkup(message) {
  const assistant = message.role === "assistant";
  const content = message.pending && !message.content
    ? (assistant && message.thinking ? "" : `<div class="thinking-row"><span class="thinking-dots"><span></span><span></span><span></span></span><span>${escapeHtml(message.status || t("searchingSources"))}</span></div>`)
    : `${renderMarkdown(message.content)}${message.pending ? '<span class="typing-cursor"></span>' : ""}`;
  return `<article class="message ${assistant ? "assistant" : "user"}" data-message-id="${message.id}"><div class="message-avatar">${assistant ? "LISA" : t("you").toUpperCase()}</div><div class="message-main"><div class="message-label">${assistant ? "LISA" : t("you")}</div>${assistant ? reasoningMarkup(message) : ""}<div class="message-content">${content}</div>${assistant ? metricsMarkup(message) : ""}${assistant && !message.pending ? citationsMarkup(message.citations, message.id) : ""}</div></article>`;
}

function bindMessageActions() {
  elements.messages.querySelectorAll(".citation-ref").forEach((button) => button.addEventListener("click", () => {
    const message = button.closest(".message");
    const card = $(`#citation-${message.dataset.messageId}-${button.dataset.citationIndex}`);
    card?.scrollIntoView({ behavior: "smooth", block: "center" });
  }));
  elements.messages.querySelectorAll(".feedback-up, .feedback-down").forEach((button) => button.addEventListener("click", () => saveFeedback(button.dataset.messageId, button.classList.contains("feedback-up") ? "up" : "down")));
  elements.messages.querySelectorAll(".copy-answer").forEach((button) => button.addEventListener("click", async () => {
    const message = state.messages.find((item) => item.id === button.dataset.messageId);
    await navigator.clipboard.writeText(message?.content || ""); toast(t("answerCopied"));
  }));
  elements.messages.querySelectorAll(".reasoning-toggle").forEach((button) => button.addEventListener("click", () => {
    const trace = button.closest(".reasoning-trace");
    const expanded = !trace.classList.contains("expanded");
    trace.classList.toggle("expanded", expanded);
    button.setAttribute("aria-expanded", String(expanded));
    const label = t(expanded ? "hideReasoning" : "showReasoning");
    button.setAttribute("aria-label", label);
    button.title = label;
  }));
}

function renderMessages({ preserveScroll = false } = {}) {
  elements.conversationTitle.textContent = displayConversationTitle(currentConversation()?.title || "New Chat");
  elements.renameButton.classList.toggle("visible", state.messages.length > 0);
  elements.main.classList.toggle("empty-state", state.messages.length === 0);
  elements.welcome.classList.toggle("hidden", state.messages.length > 0);
  elements.messages.innerHTML = state.messages.map(messageMarkup).join("");
  bindMessageActions();
  if (!preserveScroll) scrollToBottom();
}

function updatePendingMessage(message) {
  updatePendingReasoning(message);
  const node = elements.messages.querySelector(`[data-message-id="${message.id}"] .message-content`);
  if (!node) return renderMessages();
  node.innerHTML = message.content ? `${renderMarkdown(message.content)}<span class="typing-cursor"></span>` : (message.thinking ? "" : `<div class="thinking-row"><span class="thinking-dots"><span></span><span></span><span></span></span><span>${escapeHtml(message.status || t("searchingSources"))}</span></div>`);
  scrollToBottom();
}

function updatePendingReasoning(message) {
  if (!message.reasoning) return;
  const trace = elements.messages.querySelector(`[data-message-id="${message.id}"] .reasoning-trace`);
  if (!trace) return;
  const active = Boolean(message.pending && message.reasoning.active);
  const hasContent = Boolean((message.reasoning.content || "").trim());
  trace.classList.toggle("active", active);
  trace.classList.toggle("has-content", hasContent);
  const stateNode = trace.querySelector(".reasoning-state");
  const summaryNode = trace.querySelector(".reasoning-summary");
  const transcript = trace.querySelector(".reasoning-transcript");
  const button = trace.querySelector(".reasoning-toggle");
  stateNode.textContent = active
    ? t("reasoningInProgress")
    : message.reasoning.truncated
      ? t("reasoningInterrupted")
      : t("reasoningComplete", { seconds: Number(message.reasoning.seconds || 0).toFixed(1) });
  summaryNode.textContent = reasoningSummary(message);
  summaryNode.classList.toggle("shimmer", active);
  transcript.textContent = message.reasoning.content || "";
  button.disabled = !hasContent;
}

function scrollToBottom() { requestAnimationFrame(() => { elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight; }); }

async function saveFeedback(messageId, value) {
  try {
    const updated = await api(`/api/messages/${messageId}/feedback`, { method: "POST", body: JSON.stringify({ value }) });
    const message = state.messages.find((item) => item.id === messageId);
    if (message) message.feedback = updated.feedback;
    renderMessages({ preserveScroll: true }); toast(t("feedbackSaved"));
  } catch (error) { toast(error.message); }
}

function setStreaming(active) {
  state.streaming = active;
  elements.input.disabled = active;
  elements.sendButton.classList.toggle("stop", active);
  elements.thinkingToggle.disabled = active;
  elements.sendIcon.innerHTML = active
    ? '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="8" y="8" width="8" height="8" rx="1"></rect></svg>'
    : '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 19V5M6 11l6-6 6 6"></path></svg>';
  elements.sendButton.disabled = !active && !elements.input.value.trim();
}

async function ensureConversation() { if (!state.currentId) await createConversation(); return state.currentId; }

function applyThinkingMode(enabled, announce = false) {
  state.thinkingEnabled = Boolean(enabled);
  localStorage.setItem("rag-ensea-thinking", String(state.thinkingEnabled));
  elements.thinkingToggle.classList.toggle("active", state.thinkingEnabled);
  elements.thinkingToggle.setAttribute("aria-pressed", String(state.thinkingEnabled));
  const description = t(state.thinkingEnabled ? "thinkingModeOn" : "thinkingModeOff");
  elements.thinkingToggle.setAttribute("aria-label", description);
  elements.thinkingToggle.title = description;
  if (announce) toast(description);
}

async function submitQuestion(question) {
  const content = question.trim();
  if (!content || state.streaming) return;
  const conversationId = await ensureConversation();
  const thinking = state.thinkingEnabled;
  const pending = {
    id: `pending-${Date.now()}`,
    role: "assistant",
    content: "",
    citations: [],
    metrics: {},
    pending: true,
    thinking,
    reasoning: thinking ? { content: "", summary_key: "general", seconds: 0, active: true } : null,
    status: thinking ? t("deepReasoning") : t("searchingSources"),
  };
  state.messages.push({ id: `local-user-${Date.now()}`, role: "user", content, citations: [], metrics: {} }, pending);
  elements.input.value = ""; resizeInput(); setStreaming(true); renderMessages();
  state.abortController = new AbortController();
  try {
    const response = await fetch(`/api/conversations/${conversationId}/messages/stream`, { method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": state.csrfToken }, body: JSON.stringify({ content, thinking }), signal: state.abortController.signal });
    if (!response.ok) { let detail = `Le serveur a répondu ${response.status}.`; try { detail = (await response.json()).detail || detail; } catch (_error) {} throw new Error(detail); }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "route") { pending.route = event.route; pending.status = event.route === "conversation" ? t("conversationReply") : thinking ? t("deepReasoning") : t("checkingIndex", { route: event.route }); updatePendingMessage(pending); }
        else if (event.type === "reasoning") { pending.reasoning ||= { content: "", summary_key: "general", seconds: 0, active: true }; pending.reasoning.content += event.content || ""; pending.reasoning.summary_key = inferReasoningSummaryKey(pending.reasoning.content); pending.reasoning.active = true; updatePendingReasoning(pending); scrollToBottom(); }
        else if (event.type === "token") { if (pending.reasoning) pending.reasoning.active = false; pending.content += event.content; updatePendingMessage(pending); }
        else if (event.type === "done") { state.messages[state.messages.findIndex((item) => item.id === pending.id)] = event.message; renderMessages(); await loadConversations({ selectFirst: false }); renderMessages({ preserveScroll: true }); }
        else if (event.type === "error") throw new Error(event.message || event.detail);
      }
      if (done) break;
    }
  } catch (error) {
    pending.content = error.name === "AbortError" ? (pending.content || t("displayStopped")) : error.message;
    pending.pending = false; renderMessages(); toast(error.name === "AbortError" ? t("answerStopped") : error.message);
  } finally { state.abortController = null; setStreaming(false); elements.input.focus(); }
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 170)}px`;
  if (!state.streaming) elements.sendButton.disabled = !elements.input.value.trim();
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("rag-ensea-theme-v2", theme);
  document.querySelectorAll("[data-theme-choice]").forEach((button) => button.classList.toggle("active", button.dataset.themeChoice === theme));
}

function applyLanguage(language) {
  state.language = translations[language] ? language : "fr";
  document.documentElement.lang = state.language;
  localStorage.setItem("rag-ensea-language", state.language);
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-language-choice]").forEach((button) => button.classList.toggle("active", button.dataset.languageChoice === state.language));
  applyThinkingMode(state.thinkingEnabled);
  updateWelcome();
  renderRoleContent();
  renderConversations();
  renderMessages({ preserveScroll: true });
  if (elements.documentsDialog.open) loadDocuments().catch(() => {});
}

function updateWelcome() {
  if (!elements.welcomeTitle) return;
  const rawName = state.user?.display_name || state.user?.email?.split("@")[0] || "";
  const firstName = rawName.trim().split(/\s+/)[0];
  if (state.language === "en") {
    elements.welcomeTitle.textContent = firstName
      ? `Welcome back, ${firstName}. What would you like to explore?`
      : "What would you like to explore?";
    elements.welcomeSubtitle.textContent = "Search official sources, reports and ENSEA campus life.";
    elements.input.placeholder = "Write a message to the ENSEA assistant";
  } else {
    elements.welcomeTitle.textContent = firstName
      ? `Ravi de vous revoir, ${firstName}. Que souhaitez-vous explorer ?`
      : "Que souhaitez-vous explorer ?";
    elements.welcomeSubtitle.textContent = "Interrogez les sources officielles, les rapports et la vie associative de l’ENSEA.";
    elements.input.placeholder = "Écrivez un message à l’assistant ENSEA";
  }
  document.querySelectorAll("[data-suggestion]").forEach((button) => {
    button.textContent = state.language === "en" ? button.dataset.labelEn : button.dataset.labelFr;
  });
}

function renderRoleContent() {
  if (!state.user) return;
  const admin = state.user.role === "admin";
  const roleLabel = t(admin ? "adminRole" : "userRole");
  elements.toolsSubtitle.textContent = t(admin ? "toolsSubtitleAdmin" : "toolsSubtitleUser");
  elements.toolsRoleBadge.textContent = roleLabel;
  elements.toolDocumentsTitle.textContent = t("documentLibrary");
  elements.toolDocumentsDescription.textContent = t(admin ? "documentsAdminDescription" : "documentsUserDescription");
  elements.toolDocumentsAction.textContent = t(admin ? "manageLibrary" : "browseLibrary");
  elements.toolMetricsTitle.textContent = t(admin ? "globalSupervision" : "myActivity");
  elements.toolMetricsDescription.textContent = t(admin ? "activityAdminDescription" : "activityUserDescription");
  elements.toolMetricsAction.textContent = t("view");
  elements.metricsDialogKicker.textContent = t(admin ? "globalObservability" : "activity");
  elements.metricsDialogTitle.textContent = t(admin ? "globalSupervision" : "myActivity");
  elements.settingsRoleBadge.textContent = roleLabel;
  elements.settingsAccessRole.textContent = roleLabel;
  elements.settingsAccessDescription.textContent = t(admin ? "adminAccessDescription" : "userAccessDescription");
  elements.settingsDocumentsDescription.textContent = t(admin ? "adminDocumentsSetting" : "userDocumentsSetting");
  elements.documentsDialogTitle.textContent = t(admin ? "addDocuments" : "documentLibrary");
}

function passageCount(value) {
  return Number.isFinite(Number(value))
    ? `${Number(value).toLocaleString(state.language === "fr" ? "fr-FR" : "en-US")} ${t("passages")}`
    : "—";
}

async function updateHealth() {
  try {
    const { service } = await api("/api/system/status");
    elements.statusDot.className = `status-dot${service.error ? " error" : service.loaded ? " ready" : ""}`;
    elements.statusLabel.textContent = service.error ? t("pipelineError") : service.loading ? t("loading") : service.busy ? t("processingNow") : service.loaded ? t("systemReady") : t("apiAvailable");
    elements.statusDetail.textContent = service.loaded ? `${service.institutional_chunks} + ${service.facebook_chunks} passages` : t("localPipeline");
    elements.modelName.textContent = service.model || "qwen3.5:9b";
    elements.toolInstitutionalCount.textContent = passageCount(service.institutional_chunks);
    elements.toolEconomicCount.textContent = passageCount(service.economic_chunks);
    elements.toolFacebookCount.textContent = passageCount(service.facebook_chunks);
    elements.toolModelName.textContent = service.model || "qwen3.5:9b";
    const profileKey = service.performance_profile === "cpu_only" ? "cpuProfile" : service.performance_profile === "balanced_gpu" ? "sharedGpuProfile" : "optimizedProfile";
    elements.toolModelProfile.textContent = t(profileKey);
    elements.toolPipelineDot.className = `status-dot${service.error ? " error" : service.loaded ? " ready" : ""}`;
    elements.toolPipelineStatus.textContent = service.error ? t("systemError") : service.maintenance ? t("maintenanceRunning") : service.loaded ? t("systemOperational") : t("connectingSystem");
  } catch (_error) {
    elements.statusDot.className = "status-dot error"; elements.statusLabel.textContent = t("apiUnavailable"); elements.statusDetail.textContent = t("checkLocalServer");
    elements.toolPipelineDot.className = "status-dot error"; elements.toolPipelineStatus.textContent = t("systemError");
    elements.toolInstitutionalCount.textContent = "—"; elements.toolEconomicCount.textContent = "—"; elements.toolFacebookCount.textContent = "—";
  }
}

async function warmPipeline() {
  try { elements.statusLabel.textContent = t("loadingPipeline"); await api("/api/pipeline/load", { method: "POST" }); }
  catch (error) { toast(error.message); }
  finally { await updateHealth(); }
}

function selectPdf(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") return toast(t("selectPdf"));
  if (file.size > 50 * 1024 * 1024) return toast(t("fileTooLarge"));
  state.selectedFile = file;
  elements.selectedFile.hidden = false;
  elements.selectedFile.innerHTML = `<span class="document-icon">PDF</span><span><strong>${escapeHtml(file.name)}</strong><small>${formatBytes(file.size)}</small></span>`;
  if (!elements.documentTitle.value.trim()) elements.documentTitle.value = file.name.replace(/\.pdf$/i, "").replaceAll(/[_-]+/g, " ");
  elements.uploadButton.disabled = false;
}

function resetUploadForm() {
  state.selectedFile = null; elements.pdfInput.value = ""; elements.selectedFile.hidden = true; elements.selectedFile.innerHTML = "";
  elements.documentTitle.value = ""; elements.documentAuthor.value = ""; elements.documentDate.value = "";
  elements.documentCategory.value = "rapport"; elements.uploadButton.disabled = true;
}

function setProgress(progress, message, status = "running") {
  elements.uploadProgress.hidden = false;
  elements.progressValue.textContent = `${Math.round(progress)} %`;
  elements.progressFill.style.width = `${Math.max(0, Math.min(100, progress))}%`;
  elements.progressTitle.textContent = status === "completed" ? t("documentReady") : status === "error" ? t("importStopped") : t("indexing");
  elements.progressMessage.textContent = message || t("processingDocument");
  elements.uploadProgress.classList.toggle("error", status === "error");
}

function documentMarkup(document) {
  const labels = { queued: t("queued"), processing: t("indexStatus"), ready: t("ready"), error: t("error") };
  const detail = document.status === "ready" ? t("pagesPassages", { pages: document.page_count || 0, chunks: document.chunk_count || 0 }) : document.error || labels[document.status] || document.status;
  return `<article class="document-item"><span class="document-icon">PDF</span><span class="document-copy"><strong>${escapeHtml(document.title || document.filename)}</strong><small>${escapeHtml(detail)}${document.author ? ` · ${escapeHtml(document.author)}` : ""}</small></span><span class="document-status ${document.status}">${labels[document.status] || document.status}</span>${document.status === "ready" ? `<a class="document-open" href="/api/documents/${document.id}/file" target="_blank" rel="noopener">↗</a>` : ""}</article>`;
}

async function loadDocuments() {
  const payload = await api("/api/documents");
  elements.documentCount.textContent = payload.documents.length;
  elements.documentList.innerHTML = payload.documents.length ? payload.documents.map(documentMarkup).join("") : `<div class="empty-library"><strong>${t("noPdf")}</strong><span>${t("importedHere")}</span></div>`;
  const active = payload.jobs.find((job) => job.document_id && ["queued", "running"].includes(job.status));
  if (active && !state.activeJobId) { state.activeJobId = active.id; pollJob(); }
  return payload;
}

async function openDocuments({ chooseFile = false } = {}) {
  closeSidebar();
  elements.documentsDialog.showModal();
  if (chooseFile) elements.pdfInput.click();
  try { await loadDocuments(); }
  catch (error) { toast(error.message); }
}

async function uploadDocument() {
  if (!state.selectedFile || state.activeJobId) return;
  const body = new FormData();
  body.append("file", state.selectedFile);
  body.append("title", elements.documentTitle.value.trim());
  body.append("author", elements.documentAuthor.value.trim());
  body.append("document_date", elements.documentDate.value);
  body.append("category", elements.documentCategory.value);
  elements.uploadButton.disabled = true; setProgress(3, t("sendingPdf"));
  try {
    const payload = await api("/api/documents", { method: "POST", body });
    state.activeJobId = payload.job.id; setProgress(payload.job.progress || 5, payload.job.message); pollJob(); await loadDocuments();
  } catch (error) { setProgress(0, error.message, "error"); elements.uploadButton.disabled = false; toast(error.message); }
}

async function pollJob() {
  window.clearTimeout(state.jobTimer);
  if (!state.activeJobId) return;
  try {
    const job = await api(`/api/jobs/${state.activeJobId}`);
    setProgress(job.progress, job.message, job.status);
    if (job.status === "completed") {
      state.activeJobId = null; resetUploadForm(); await loadDocuments(); toast(t("pdfIndexed")); await updateHealth(); return;
    }
    if (job.status === "error") { state.activeJobId = null; elements.uploadButton.disabled = !state.selectedFile; toast(job.error || t("indexingFailed")); await loadDocuments(); return; }
    state.jobTimer = window.setTimeout(pollJob, 900);
  } catch (error) { state.jobTimer = window.setTimeout(pollJob, 1800); }
}

async function openMetrics() {
  try {
    const { application: metrics } = await api("/api/metrics");
    const totalFeedback = metrics.positive_feedback + metrics.negative_feedback;
    const cards = [[t("conversations"), metrics.conversations], [t("questions"), metrics.questions], [t("pdfDocuments"), metrics.indexed_documents || 0], [t("addedPassages"), metrics.uploaded_chunks || 0], [t("averageTime"), `${Number(metrics.avg_seconds).toFixed(1)} s`], [t("firstToken"), `${Number(metrics.avg_first_token_seconds || 0).toFixed(1)} s`], [t("throughput"), `${Number(metrics.avg_tps || 0).toFixed(1)} tok/s`], [t("satisfaction"), totalFeedback ? `${Math.round(metrics.positive_feedback / totalFeedback * 100)} %` : "—"]];
    elements.metricGrid.innerHTML = cards.map(([label, value]) => `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
    const routes = metrics.routes || {}; const total = Math.max(1, Object.values(routes).reduce((sum, value) => sum + value, 0));
    elements.routeBreakdown.innerHTML = `<h3>${t("sourceBreakdown")}</h3>${Object.entries(routes).map(([route, count]) => `<div class="route-row"><span>${escapeHtml(route)}</span><span class="route-track"><span class="route-fill" style="width:${count / total * 100}%"></span></span><strong>${count}</strong></div>`).join("") || `<p class="dialog-note">${t("noRecordedAnswer")}</p>`}`;
    closeSidebar();
    if (elements.settingsDialog.open) elements.settingsDialog.close();
    elements.metricsDialog.showModal();
  } catch (error) { toast(error.message); }
}

function showAuth(mode = "login") {
  if (typeof mode === "boolean") mode = mode ? "setup" : "login";
  state.user = null; state.csrfToken = null; state.setupRequired = mode === "setup"; state.authMode = mode;
  elements.authScreen.hidden = false; elements.shell.setAttribute("aria-hidden", "true");
  const copy = {
    setup: ["Créer le compte administrateur", "Initialisez l’espace sécurisé. Les données existantes seront rattachées à ce compte.", "Initialiser l’application"],
    login: ["Connexion", "Retrouvez vos conversations et les sources documentaires de l’ENSEA.", "Se connecter"],
    register: ["Créer un compte", "Créez votre espace personnel pour conserver vos conversations et vos retours.", "Créer mon compte"],
    forgot: ["Récupérer votre compte", "Indiquez votre adresse. Si elle correspond à un compte actif, nous vous enverrons un lien sécurisé.", "Envoyer le lien"],
    reset: ["Nouveau mot de passe", "Choisissez un nouveau mot de passe pour votre compte.", "Mettre à jour le mot de passe"],
  }[mode] || ["Connexion", "", "Se connecter"];
  elements.authTitle.textContent = copy[0]; elements.authDescription.textContent = copy[1]; elements.authSubmit.textContent = copy[2];
  const needsName = ["setup", "register"].includes(mode);
  const needsEmail = ["setup", "login", "register", "forgot"].includes(mode);
  const needsPassword = ["setup", "login", "register", "reset"].includes(mode);
  const needsConfirm = ["register", "reset"].includes(mode);
  elements.authNameField.hidden = !needsName; elements.authName.required = needsName;
  elements.authEmailField.hidden = !needsEmail; elements.authEmail.required = needsEmail;
  elements.authPasswordField.hidden = !needsPassword; elements.authPassword.required = needsPassword;
  elements.authConfirmField.hidden = !needsConfirm; elements.authConfirm.required = needsConfirm;
  elements.authPassword.autocomplete = ["register", "reset", "setup"].includes(mode) ? "new-password" : "current-password";
  elements.authLinks.hidden = mode !== "login";
  elements.authRegister.hidden = !state.authCapabilities.registration_enabled;
  elements.authForgot.hidden = !state.authCapabilities.password_reset_enabled;
  elements.authGuest.hidden = mode !== "login" || !state.authCapabilities.guest_enabled;
  elements.authBack.hidden = ["login", "setup"].includes(mode);
  elements.authError.hidden = true;
}

function applyCurrentUser(payload) {
  state.user = payload.user; state.csrfToken = payload.csrf_token;
  elements.authScreen.hidden = true; elements.shell.removeAttribute("aria-hidden");
  const name = state.user.display_name || state.user.email;
  elements.accountName.textContent = name;
  elements.accountRole.textContent = state.user.role === "admin" ? "Administrateur" : state.user.account_type === "guest" ? "Invité" : "Utilisateur";
  elements.accountAvatar.textContent = name.slice(0, 1).toUpperCase();
  elements.settingsAccountName.textContent = name;
  elements.settingsAccountEmail.textContent = state.user.account_type === "guest" ? "Session temporaire non associée à une adresse" : state.user.email;
  elements.adminButton.hidden = state.user.role !== "admin";
  elements.attachButton.hidden = state.user.role !== "admin";
  document.body.classList.toggle("role-admin", state.user.role === "admin");
  document.body.classList.toggle("role-user", state.user.role !== "admin");
  updateWelcome();
  renderRoleContent();
}

async function submitAuth(event) {
  event.preventDefault(); elements.authSubmit.disabled = true; elements.authError.hidden = true;
  try {
    if (["register", "reset"].includes(state.authMode) && elements.authPassword.value !== elements.authConfirm.value) throw new Error("Les deux mots de passe ne correspondent pas.");
    if (state.authMode === "forgot") {
      const result = await api("/api/auth/password/forgot", { method: "POST", body: JSON.stringify({ email: elements.authEmail.value.trim() }) });
      if (result.preview_url) { state.resetToken = new URL(result.preview_url).searchParams.get("reset_token"); showAuth("reset"); toast("Mode test : le lien sécurisé a été ouvert directement."); }
      else { showAuth("login"); toast(result.message); }
      return;
    }
    if (state.authMode === "reset") {
      const result = await api("/api/auth/password/reset", { method: "POST", body: JSON.stringify({ token: state.resetToken, password: elements.authPassword.value }) });
      window.history.replaceState({}, "", window.location.pathname); state.resetToken = ""; elements.authForm.reset(); showAuth("login"); toast(result.message); return;
    }
    const payload = { email: elements.authEmail.value.trim(), password: elements.authPassword.value };
    if (["setup", "register"].includes(state.authMode)) payload.display_name = elements.authName.value.trim();
    const endpoint = state.authMode === "setup" ? "/api/auth/setup" : state.authMode === "register" ? "/api/auth/register" : "/api/auth/login";
    const result = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
    applyCurrentUser(result); elements.authForm.reset(); await initializeWorkspace();
  } catch (error) { elements.authError.textContent = error.message; elements.authError.hidden = false; }
  finally { elements.authSubmit.disabled = false; }
}

async function continueAsGuest() {
  elements.authGuest.disabled = true; elements.authError.hidden = true;
  try { const result = await api("/api/auth/guest", { method: "POST" }); applyCurrentUser(result); await initializeWorkspace(); }
  catch (error) { elements.authError.textContent = error.message; elements.authError.hidden = false; }
  finally { elements.authGuest.disabled = false; }
}

async function logout() {
  try { await api("/api/auth/logout", { method: "POST" }); } catch (_error) {}
  elements.settingsDialog.close(); state.conversations = []; state.messages = []; state.currentId = null;
  showAuth(false);
}

async function openSessions() {
  try {
    const sessions = await api("/api/auth/sessions");
    elements.sessionsList.innerHTML = sessions.map((session) => `<article class="admin-list-row"><div><strong>${session.current ? "Cette session" : "Session"}</strong><span>${escapeHtml(session.user_agent || "Navigateur inconnu")} · ${relativeDate(session.last_seen_at)}</span></div>${session.current ? '<span class="status-badge">Active</span>' : `<button class="danger-button compact" data-revoke-session="${session.id}">Révoquer</button>`}</article>`).join("");
    elements.sessionsDialog.showModal();
  } catch (error) { toast(error.message); }
}

function adminMetricMarkup(data) {
  const metrics = data.metrics || {}; const pipeline = data.pipeline || {};
  const values = [["Utilisateurs", data.users?.length || 0], ["Conversations", metrics.conversations || 0], ["Questions", metrics.questions || 0], ["Temps moyen", `${Number(metrics.avg_seconds || 0).toFixed(1)} s`], ["Premier mot", `${Number(metrics.avg_first_token_seconds || 0).toFixed(1)} s`], ["Attente", `${Number(metrics.avg_queue_seconds || 0).toFixed(2)} s`], ["Débit", `${Number(metrics.avg_tps || 0).toFixed(1)} tok/s`], ["Profil", pipeline.performance_profile || "—"], ["Pipeline", pipeline.maintenance ? "Maintenance" : pipeline.loaded ? "Chargé" : "Au repos"]];
  return values.map(([label, value]) => `<article><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`).join("");
}

function renderSecurityOverview(data) {
  const identity = data.identity || {};
  const values = [["Comptes", identity.registered || 0], ["Invités", identity.guests || 0], ["Sessions actives", identity.active_sessions || 0], ["Inscriptions · 24 h", identity.registrations_24h || 0], ["Récupérations", identity.recovery_requests || 0]];
  elements.securityMetrics.innerHTML = values.map(([label, value]) => `<article><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`).join("");
  elements.securitySessionList.innerHTML = (data.security_sessions || []).map((session) => `<article class="admin-list-row"><span class="account-avatar">${session.account_type === "guest" ? "G" : escapeHtml((session.display_name || "U").slice(0, 1).toUpperCase())}</span><div><strong>${escapeHtml(session.account_type === "guest" ? "Session invitée" : session.display_name)}</strong><span>${escapeHtml(session.user_agent || "Navigateur inconnu")} · ${escapeHtml(session.ip_address || "IP inconnue")} · ${relativeDate(session.last_seen_at)}</span></div>${session.revoked_at ? '<span class="document-status error">Révoquée</span>' : `<button class="danger-button compact" data-admin-revoke-session="${session.id}">Révoquer</button>`}</article>`).join("") || '<p class="empty-admin">Aucune session.</p>';
  const labels = { "auth.login": "Connexion", "auth.logout": "Déconnexion", "auth.registration": "Inscription", "auth.guest_started": "Accès invité", "auth.password_reset_requested": "Récupération demandée", "auth.password_reset_completed": "Mot de passe modifié", "user.created": "Compte créé", "user.updated": "Compte modifié" };
  elements.securityAuditList.innerHTML = (data.audit_events || []).map((event) => `<article class="admin-list-row"><div><strong>${escapeHtml(labels[event.action] || event.action)}</strong><span>${escapeHtml(event.actor_email || "Système")} · ${relativeDate(event.created_at)}</span></div></article>`).join("") || '<p class="empty-admin">Aucun événement.</p>';
}

function selectedFacebookPages() {
  return [...elements.facebookPageOptions.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
}

function updateFacebookEstimate() {
  const pages = selectedFacebookPages().length;
  const perPage = Math.max(1, Math.min(100, Number(elements.facebookMaxPosts.value) || 0));
  const activeJob = (state.adminData?.jobs || []).find((job) =>
    (job.id === state.activeAdminJobId || ["collect_facebook", "full_refresh"].includes(job.kind))
      && ["queued", "running"].includes(job.status)
  );
  elements.facebookSelectionSummary.textContent = `${pages} page${pages > 1 ? "s" : ""} sélectionnée${pages > 1 ? "s" : ""}`;
  elements.facebookEstimate.textContent = `Jusqu’à ${pages * perPage} publication${pages * perPage > 1 ? "s" : ""} au total`;
  elements.facebookLaunch.disabled = pages === 0 || Boolean(activeJob);
}

function renderFacebookConfiguration(data) {
  const facebook = data.facebook || {}; const pages = facebook.pages || {};
  const signature = JSON.stringify(pages);
  if (signature !== state.facebookPagesSignature) {
    state.facebookPagesSignature = signature;
    elements.facebookPageOptions.innerHTML = Object.entries(pages).map(([key, url]) => {
      const label = key.replace(/_page$/i, "").replaceAll("_", " ");
      return `<label class="facebook-page-option"><input type="checkbox" value="${escapeHtml(key)}" checked><span class="facebook-page-check">✓</span><span><strong>${escapeHtml(label)}</strong><small>${escapeHtml(url)}</small></span></label>`;
    }).join("");
    elements.facebookMaxPosts.value = facebook.default_max_posts_per_page || 30;
    elements.scheduleFacebookPages.innerHTML = Object.keys(pages).map((key) => `<label><input type="checkbox" value="${escapeHtml(key)}" checked><span>${escapeHtml(key.replace(/_page$/i, "").replaceAll("_", " "))}</span></label>`).join("");
    elements.scheduleFacebookMaxPosts.value = facebook.default_max_posts_per_page || 30;
    updateFacebookEstimate();
  }
  const configured = Boolean(facebook.token_configured);
  elements.facebookTokenState.textContent = configured ? "Jeton prêt" : "Jeton requis";
  elements.facebookTokenState.className = `credential-state ${configured ? "ready" : "missing"}`;
  elements.facebookToken.placeholder = configured ? "Laisser vide pour utiliser le jeton mémorisé" : "apify_api_…";
  elements.facebookTokenHint.textContent = configured
    ? "Un jeton est déjà disponible sur ce serveur. Saisissez-en un seulement pour le remplacer."
    : "Le jeton n’est pas enregistré dans les tâches. Vous pouvez le conserver uniquement en mémoire serveur.";

  const jobs = data.jobs || [];
  const job = jobs.find((item) => item.id === state.activeAdminJobId)
    || jobs.find((item) => ["collect_facebook", "full_refresh"].includes(item.kind) && ["queued", "running"].includes(item.status));
  if (!job) return;
  state.activeAdminJobId = job.id;
  elements.facebookJobStatus.hidden = false;
  elements.facebookJobStatus.dataset.status = job.status;
  elements.facebookJobTitle.textContent = job.status === "completed" ? "Collecte terminée" : job.status === "error" ? "Collecte interrompue" : "Collecte Facebook en cours";
  elements.facebookJobMessage.textContent = job.error || job.message || "En attente";
  elements.facebookJobPercent.textContent = `${job.progress || 0} %`;
  elements.facebookJobFill.style.width = `${job.progress || 0}%`;
  elements.facebookJobDetail.textContent = job.status === "completed"
    ? "Les publications brutes ont été enregistrées. Vous pouvez ensuite nettoyer et reconstruire l’index Facebook."
    : job.status === "error"
      ? "Corrigez le paramètre indiqué puis relancez la collecte."
      : "Vous pouvez changer d’onglet : cette progression continuera à se mettre à jour.";
  elements.facebookLaunch.disabled = ["queued", "running"].includes(job.status);
}

function operationDefinition(kind, data = {}) {
  const operations = data.operations || {};
  const institutional = operations.institutional || {};
  const localDocuments = operations.local_documents || {};
  const cleaning = operations.cleaning || {};
  const chunking = operations.chunking || {};
  const embeddings = operations.embeddings || {};
  const indexing = operations.indexing || {};
  const corpora = (value) => (value || ["institutionnel", "économique", "Facebook"])
    .map((name) => `<span class="operation-chip">${escapeHtml(name)}</span>`).join("");
  const definitions = {
    collect_institutional: {
      kicker: "COLLECTE INSTITUTIONNELLE", title: "Configurer la collecte du site ENSEA",
      description: "Définissez le périmètre du crawl avant de contacter le site officiel.",
      launch: "Lancer la collecte", impact: "Les fichiers bruts seront mis à jour. L’index utilisé par les conversations ne change qu’après une reconstruction.",
      parameters: `
        <div class="parameter-row"><div><strong>Point de départ</strong><span>Adresse officielle utilisée pour découvrir les pages.</span></div><code class="operation-value">${escapeHtml(institutional.base_url || "—")}</code></div>
        <label class="parameter-row operation-control"><div><strong>Limite de pages</strong><span>0 signifie : parcourir toutes les pages découvertes.</span></div><input id="institutional-max-pages" type="number" min="0" max="20000" step="1" value="${Number(institutional.max_pages || 0)}"></label>
        <label class="parameter-row operation-control"><div><strong>Téléchargements simultanés</strong><span>4 est un bon équilibre entre rapidité et charge du site.</span></div><input id="institutional-concurrency" type="number" min="1" max="12" step="1" value="${Number(institutional.max_concurrency || 4)}"></label>
        <label class="parameter-row operation-switch"><div><strong>Inclure les PDF institutionnels</strong><span>Télécharge et extrait aussi les PDF rencontrés pendant le crawl.</span></div><input id="institutional-download-pdfs" type="checkbox" ${institutional.download_pdfs === false ? "" : "checked"}><span aria-hidden="true"></span></label>`,
    },
    import_local_documents: {
      kicker: "BIBLIOTHÈQUE LOCALE", title: "Synchroniser les PDF locaux",
      description: "Contrôlez la bibliothèque détectée avant l’extraction des rapports.",
      launch: "Importer les PDF", impact: "Les PDF déjà identiques seront réutilisés. Les documents scannés sans texte seront signalés pour OCR.",
      parameters: `
        <div class="parameter-row"><div><strong>Dossier analysé</strong><span>Emplacement configuré pour les rapports ANSTAT et autres documents locaux.</span></div><code class="operation-value">${escapeHtml(localDocuments.source_dir || "—")}</code></div>
        <div class="parameter-row"><div><strong>PDF détectés</strong><span>Nombre de fichiers actuellement présents dans ce dossier et ses sous-dossiers.</span></div><strong class="operation-count">${Number(localDocuments.pdf_count || 0)}</strong></div>
        <label class="parameter-row operation-switch"><div><strong>Continuer si certains PDF nécessitent un OCR</strong><span>Les fichiers exploitables sont importés et les échecs restent visibles dans le rapport.</span></div><input id="pdf-continue-errors" type="checkbox" checked><span aria-hidden="true"></span></label>`,
    },
    clean: {
      kicker: "PRÉPARATION DES DONNÉES", title: "Nettoyer toutes les sources",
      description: "Normalise les textes avant le découpage en passages.",
      launch: "Lancer le nettoyage", impact: "Le dossier nettoyé est régénéré. Les sources brutes et les PDF originaux restent intacts.",
      parameters: `<div class="parameter-row"><div><strong>Corpus concernés</strong><span>Les trois familles sont traitées avec leurs règles adaptées.</span></div><div class="operation-chips">${corpora(cleaning.corpora)}</div></div><div class="parameter-row"><div><strong>Texte minimal conservé</strong><span>Les contenus trop courts ou sans valeur documentaire sont écartés.</span></div><strong class="operation-count">${Number(cleaning.min_content_length || 80)} caractères</strong></div>`,
    },
    chunk: {
      kicker: "DÉCOUPAGE DOCUMENTAIRE", title: "Reconstruire les passages",
      description: "Recrée la structure parent-enfant transmise à la recherche et au modèle.",
      launch: "Reconstruire les chunks", impact: "Les anciens chunks sont remplacés. Il faudra ensuite recalculer les embeddings puis les index.",
      parameters: `<div class="operation-summary-grid"><article><span>Passage enfant</span><strong>${Number(chunking.child_tokens || 400)} tokens</strong><small>Utilisé pour retrouver précisément l’information.</small></article><article><span>Contexte parent</span><strong>${Number(chunking.parent_tokens || 1800)} tokens</strong><small>Transmis au modèle pour répondre avec assez de contexte.</small></article><article><span>Chevauchement</span><strong>${Number(chunking.overlap_percent || 15)} %</strong><small>Évite de couper une information entre deux passages.</small></article></div>`,
    },
    embed: {
      kicker: "REPRÉSENTATION VECTORIELLE", title: "Recalculer tous les embeddings",
      description: "Encode chaque passage avec le modèle dense configuré.",
      launch: "Recalculer les embeddings", impact: "Cette opération sollicite fortement le GPU et suspend temporairement les réponses. Elle peut prendre plusieurs minutes.",
      parameters: `<div class="operation-summary-grid"><article><span>Modèle</span><strong>${escapeHtml(embeddings.model || "—")}</strong><small>Embeddings multilingues et recherche dense.</small></article><article><span>Calcul</span><strong>${escapeHtml(String(embeddings.device || "—").toUpperCase())} · ${escapeHtml(embeddings.precision || "—")}</strong><small>Profil utilisé pendant l’indexation hors ligne.</small></article><article><span>Lot</span><strong>${Number(embeddings.batch_size || 16)} passages</strong><small>Nombre de passages encodés ensemble.</small></article></div>`,
    },
    index: {
      kicker: "INDEX DE RECHERCHE", title: "Reconstruire LanceDB et BM25",
      description: "Synchronise la recherche sémantique et la recherche lexicale.",
      launch: "Reconstruire les index", impact: "Les chunks et embeddings doivent déjà correspondre. Une incohérence arrêtera l’opération sans masquer l’erreur.",
      parameters: `<div class="parameter-row"><div><strong>Corpus indexés</strong><span>Chaque corpus conserve son index et son routage.</span></div><div class="operation-chips">${corpora(indexing.corpora)}</div></div><div class="operation-summary-grid two"><article><span>Recherche dense</span><strong>${escapeHtml(indexing.dense || "LanceDB")}</strong><small>Retrouve les passages proches par le sens.</small></article><article><span>Recherche lexicale</span><strong>${escapeHtml(indexing.lexical || "BM25")}</strong><small>Retrouve les noms, dates et termes exacts.</small></article></div>`,
    },
    rebuild: {
      kicker: "MAINTENANCE COMPLÈTE", title: "Reconstruire le RAG",
      description: "Exécute les quatre étapes nécessaires à partir des sources déjà collectées.",
      launch: "Lancer la reconstruction", impact: "Le GPU sera réservé et les réponses seront temporairement suspendues. Aucune nouvelle page web ou publication Facebook ne sera collectée.",
      parameters: `<ol class="operation-sequence"><li><span>1</span><div><strong>Nettoyage</strong><small>Normalisation des corpus.</small></div></li><li><span>2</span><div><strong>Chunking</strong><small>Passages parent-enfant.</small></div></li><li><span>3</span><div><strong>Embeddings</strong><small>Vecteurs BGE-M3.</small></div></li><li><span>4</span><div><strong>Indexation</strong><small>LanceDB et BM25.</small></div></li></ol>`,
    },
  };
  return definitions[kind];
}

function renderOperationJob(data) {
  if (!state.operationJobKind) return;
  const jobs = data.jobs || [];
  const job = jobs.find((item) => item.id === state.activeOperationJobId)
    || jobs.find((item) => item.kind === state.operationJobKind && ["queued", "running"].includes(item.status));
  if (!job) return;
  state.activeOperationJobId = job.id;
  elements.operationJobStatus.hidden = false;
  elements.operationJobStatus.dataset.status = job.status;
  elements.operationJobTitle.textContent = data.tasks?.[job.kind] || "Traitement";
  elements.operationJobMessage.textContent = job.error || job.message || "En attente";
  elements.operationJobPercent.textContent = `${job.progress || 0} %`;
  elements.operationJobFill.style.width = `${job.progress || 0}%`;
  elements.operationJobDetail.textContent = job.status === "completed"
    ? "L’opération est terminée et le pipeline a été rechargé."
    : job.status === "error"
      ? "Le traitement s’est arrêté. Le message ci-dessus indique la cause à corriger."
      : "Vous pouvez changer d’onglet : le traitement continue en arrière-plan.";
  elements.operationLaunch.disabled = ["queued", "running"].includes(job.status);
}

function openOperationConfigurator(kind) {
  const definition = operationDefinition(kind, state.adminData || {});
  if (!definition) return;
  closeFacebookCollector();
  state.operationJobKind = kind;
  state.activeOperationJobId = null;
  elements.operationKicker.textContent = definition.kicker;
  elements.operationTitle.textContent = definition.title;
  elements.operationDescription.textContent = definition.description;
  elements.operationParameters.innerHTML = definition.parameters;
  elements.operationImpact.innerHTML = `<span aria-hidden="true">i</span><div><strong>À savoir avant de lancer</strong><p>${escapeHtml(definition.impact)}</p></div>`;
  elements.operationLaunch.textContent = definition.launch;
  elements.operationLaunch.disabled = false;
  elements.operationJobStatus.hidden = true;
  elements.operationConfigurator.hidden = false;
  requestAnimationFrame(() => elements.operationConfigurator.scrollIntoView({ behavior: "smooth", block: "start" }));
}

function closeOperationConfigurator() {
  elements.operationConfigurator.hidden = true;
}

function operationPayload(kind) {
  if (kind === "collect_institutional") return {
    max_pages: Number(elements.operationParameters.querySelector("#institutional-max-pages")?.value || 0),
    max_concurrency: Number(elements.operationParameters.querySelector("#institutional-concurrency")?.value || 4),
    download_pdfs: Boolean(elements.operationParameters.querySelector("#institutional-download-pdfs")?.checked),
  };
  if (kind === "import_local_documents") return {
    continue_on_pdf_errors: Boolean(elements.operationParameters.querySelector("#pdf-continue-errors")?.checked),
  };
  return {};
}

async function submitConfiguredOperation(event) {
  event.preventDefault();
  const kind = state.operationJobKind;
  if (!kind) return;
  const payload = operationPayload(kind);
  if (kind === "collect_institutional" && (!Number.isInteger(payload.max_pages) || payload.max_pages < 0 || payload.max_pages > 20000)) { toast("Choisissez une limite comprise entre 0 et 20 000 pages."); return; }
  if (kind === "collect_institutional" && (!Number.isInteger(payload.max_concurrency) || payload.max_concurrency < 1 || payload.max_concurrency > 12)) { toast("Choisissez entre 1 et 12 téléchargements simultanés."); return; }
  elements.operationLaunch.disabled = true;
  const originalLabel = elements.operationLaunch.textContent;
  elements.operationLaunch.textContent = "Ajout à la file…";
  try {
    const job = await api("/api/admin/jobs", { method: "POST", body: JSON.stringify({ kind, payload }) });
    state.activeOperationJobId = job.id;
    elements.operationJobStatus.hidden = false;
    elements.operationJobMessage.textContent = "L’opération a été ajoutée à la file.";
    elements.operationJobPercent.textContent = "0 %";
    elements.operationJobFill.style.width = "0%";
    toast("Traitement lancé.");
    await loadAdmin();
  } catch (error) { toast(error.message); elements.operationLaunch.disabled = false; }
  finally { elements.operationLaunch.textContent = originalLabel; }
}

function openFacebookCollector(kind = "collect_facebook") {
  closeOperationConfigurator();
  state.facebookJobKind = kind;
  const fullRefresh = kind === "full_refresh";
  elements.facebookCollectorTitle.textContent = fullRefresh ? "Configurer le cycle documentaire complet" : "Configurer la collecte Facebook";
  elements.facebookCollectorDescription.textContent = fullRefresh
    ? "La collecte des pages choisies sera suivie du nettoyage, du chunking, des embeddings et de l’indexation."
    : "Choisissez exactement les pages et le volume à récupérer avant de lancer Apify.";
  elements.facebookLaunch.textContent = fullRefresh ? "Lancer le cycle complet" : "Lancer la collecte";
  elements.facebookCollector.hidden = false;
  renderFacebookConfiguration(state.adminData || {});
  requestAnimationFrame(() => {
    elements.facebookCollector.scrollIntoView({ behavior: "smooth", block: "start" });
    if (!state.adminData?.facebook?.token_configured) elements.facebookToken.focus({ preventScroll: true });
  });
}

function closeFacebookCollector() {
  elements.facebookCollector.hidden = true;
  elements.facebookToken.value = "";
}

function updateScheduleFields() {
  elements.scheduleFacebookConfig.hidden = !["collect_facebook", "full_refresh"].includes(elements.scheduleKind.value);
}

function renderAdmin(data) {
  state.adminData = data; elements.adminMetrics.innerHTML = adminMetricMarkup(data);
  const pipeline = data.pipeline || {};
  const offlineMode = Boolean(data.deployment?.offline_mode);
  elements.adminHealth.textContent = pipeline.maintenance ? "Maintenance en cours" : pipeline.busy ? "Moteur occupé" : offlineMode ? "Système disponible · hors ligne" : "Système disponible";
  elements.adminHealth.className = `admin-health${pipeline.maintenance || pipeline.error ? " warning" : ""}`;
  const networkKinds = new Set(["collect_institutional", "collect_facebook", "full_refresh"]);
  elements.operationGrid.querySelectorAll("[data-job-kind]").forEach((button) => {
    const blocked = offlineMode && networkKinds.has(button.dataset.jobKind);
    button.disabled = blocked;
    button.title = blocked ? "Connexion réseau désactivée par le mode hors ligne" : "";
  });
  [...elements.scheduleKind.options].forEach((option) => {
    option.disabled = offlineMode && networkKinds.has(option.value);
  });
  if (elements.scheduleKind.selectedOptions[0]?.disabled) {
    elements.scheduleKind.value = "import_local_documents";
    updateScheduleFields();
  }
  const settings = data.settings || {}; const models = data.models || {};
  const generationModels = (models.generation || []).filter((value, index, array) => array.indexOf(value) === index);
  elements.generationModel.innerHTML = generationModels.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
  const savedGenerationModel = settings.generation_model;
  elements.generationModel.value = generationModels.includes(savedGenerationModel)
    ? savedGenerationModel
    : (pipeline.model || generationModels[0] || "qwen3.5:9b");
  elements.performanceProfile.value = pipeline.performance_profile === "cpu_only" ? "cpu_only" : (settings.performance_profile || pipeline.performance_profile || "generation_priority_8gb");
  elements.modelInventory.innerHTML = `<article><span>Embeddings</span><strong>${escapeHtml(models.embedding || "—")}</strong><small>Index dense · ${escapeHtml(pipeline.embedding_device || "non chargé")}</small></article><article><span>Reranker</span><strong>${escapeHtml(models.reranker || "—")}</strong><small>Cross-encoder · ${escapeHtml(pipeline.reranker_device || "non chargé")}</small></article><article><span>Génération</span><strong>${escapeHtml(elements.generationModel.value)}</strong><small>Ollama · fenêtre 4 096 tokens</small></article>`;
  renderFacebookConfiguration(data);
  renderOperationJob(data);
  elements.userList.innerHTML = (data.users || []).map((user) => `<article class="admin-list-row" data-user-id="${user.id}"><span class="account-avatar">${user.account_type === "guest" ? "G" : escapeHtml((user.display_name || user.email).slice(0, 1).toUpperCase())}</span><div><strong>${escapeHtml(user.account_type === "guest" ? "Invité temporaire" : user.display_name)}</strong><span>${user.account_type === "guest" ? "Session sans compte" : escapeHtml(user.email)} · ${user.last_login_at ? `vu ${relativeDate(user.last_login_at)}` : "jamais connecté"}</span></div><select data-user-role ${user.id === state.user.id || user.account_type === "guest" ? "disabled" : ""}><option value="user" ${user.role === "user" ? "selected" : ""}>Utilisateur</option><option value="admin" ${user.role === "admin" ? "selected" : ""}>Administrateur</option></select><button class="${user.is_active ? "secondary-button" : "primary-button"} compact" data-toggle-user ${user.id === state.user.id ? "disabled" : ""}>${user.is_active ? "Désactiver" : "Réactiver"}</button></article>`).join("");
  renderSecurityOverview(data);
  elements.scheduleList.innerHTML = (data.schedules || []).map((schedule) => `<article class="admin-list-row" data-schedule-id="${schedule.id}"><div><strong>${escapeHtml(schedule.name)}</strong><span>${escapeHtml(data.tasks?.[schedule.task_kind] || schedule.task_kind)} · prochaine exécution ${relativeDate(schedule.next_run_at)}</span></div><label class="mini-switch"><input type="checkbox" data-toggle-schedule ${schedule.enabled ? "checked" : ""}><span></span></label><button class="danger-button compact" data-delete-schedule>Supprimer</button></article>`).join("") || '<p class="empty-admin">Aucune planification.</p>';
  const statusLabels = { queued: "En attente", running: "En cours", completed: "Terminé", error: "Échec" };
  elements.jobList.innerHTML = (data.jobs || []).map((job) => `<article class="admin-list-row job-row"><span class="job-state ${job.status}">${job.progress}%</span><div><strong>${escapeHtml(data.tasks?.[job.kind] || job.kind)}</strong><span>${escapeHtml(job.message || "En attente")} · ${relativeDate(job.created_at)}</span><div class="job-progress-track"><span class="${job.status}" style="width:${Number(job.progress || 0)}%"></span></div>${job.error ? `<small class="job-error">${escapeHtml(job.error)}</small>` : ""}</div><span class="document-status ${job.status}">${statusLabels[job.status] || escapeHtml(job.status)}</span></article>`).join("") || '<p class="empty-admin">Aucune activité.</p>';
}

async function synchronizeAdminJobCompletion(data) {
  let completedNow = false;
  for (const job of data.jobs || []) {
    const previous = state.adminJobStatuses.get(job.id);
    if (["queued", "running"].includes(previous) && ["completed", "error"].includes(job.status)) {
      completedNow = true;
    }
    state.adminJobStatuses.set(job.id, job.status);
  }
  if (completedNow) await updateHealth();
}

async function loadAdmin({ silent = false } = {}) {
  if (!state.user || state.user.role !== "admin") return;
  try {
    const data = await api("/api/admin/overview");
    renderAdmin(data);
    await synchronizeAdminJobCompletion(data);
  }
  catch (error) { if (!silent) toast(error.message); }
}

function selectAdminTab(name) {
  document.querySelectorAll("[data-admin-tab]").forEach((item) => item.classList.toggle("active", item.dataset.adminTab === name));
  document.querySelectorAll("[data-admin-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.adminPanel === name));
}

async function openAdmin(tab = "operations") {
  if (!state.user || state.user.role !== "admin") return;
  selectAdminTab(tab);
  closeSidebar(); elements.adminDialog.showModal(); await loadAdmin();
  window.clearInterval(state.adminTimer); state.adminTimer = window.setInterval(() => loadAdmin({ silent: true }), 2500);
}

function closeAdmin() { window.clearInterval(state.adminTimer); state.adminTimer = null; elements.adminDialog.close(); updateHealth(); }

async function launchAdminJob(kind) {
  if (state.adminData?.deployment?.offline_mode && ["collect_institutional", "collect_facebook", "full_refresh"].includes(kind)) {
    toast("Mode hors ligne actif : les collectes réseau sont désactivées.");
    return;
  }
  if (["collect_facebook", "full_refresh"].includes(kind)) { openFacebookCollector(kind); return; }
  openOperationConfigurator(kind);
}

async function submitFacebookCollection(event) {
  event.preventDefault();
  const pageKeys = selectedFacebookPages();
  const maxPosts = Number(elements.facebookMaxPosts.value);
  const token = elements.facebookToken.value.trim();
  if (!pageKeys.length) { toast("Sélectionnez au moins une page Facebook."); return; }
  if (!Number.isInteger(maxPosts) || maxPosts < 1 || maxPosts > 100) { toast("Choisissez entre 1 et 100 publications par page."); elements.facebookMaxPosts.focus(); return; }
  if (!state.adminData?.facebook?.token_configured && !token) { toast("Saisissez votre jeton Apify pour lancer la collecte."); elements.facebookToken.focus(); return; }
  elements.facebookLaunch.disabled = true;
  elements.facebookLaunch.textContent = "Ajout à la file…";
  try {
    const job = await api("/api/admin/jobs", {
      method: "POST",
      body: JSON.stringify({
        kind: state.facebookJobKind,
        payload: { page_keys: pageKeys, max_posts_per_page: maxPosts },
        apify_token: token || null,
        remember_apify_token: Boolean(elements.facebookRememberToken.checked),
      }),
    });
    state.activeAdminJobId = job.id;
    elements.facebookToken.value = "";
    elements.facebookJobStatus.hidden = false;
    elements.facebookJobMessage.textContent = "La collecte a été ajoutée à la file.";
    elements.facebookJobPercent.textContent = "0 %";
    elements.facebookJobFill.style.width = "0%";
    toast(state.facebookJobKind === "full_refresh" ? "Cycle documentaire lancé." : "Collecte Facebook lancée.");
    await loadAdmin();
  } catch (error) { toast(error.message); elements.facebookLaunch.disabled = false; }
  finally { elements.facebookLaunch.textContent = state.facebookJobKind === "full_refresh" ? "Lancer le cycle complet" : "Lancer la collecte"; }
}

elements.newChat.addEventListener("click", startNewChat);
elements.collapsedNewChat.addEventListener("click", startNewChat);
elements.launchButton.addEventListener("click", () => showView("launch"));
elements.search.addEventListener("input", renderConversations);
elements.renameButton.addEventListener("click", renameConversation);
elements.menuButton.addEventListener("click", openSidebar); elements.sidebarClose.addEventListener("click", collapseSidebar); elements.sidebarBackdrop.addEventListener("click", closeSidebar);
elements.input.addEventListener("input", resizeInput);
document.querySelectorAll("[data-suggestion]").forEach((button) => button.addEventListener("click", () => {
  elements.input.value = state.language === "en" ? button.dataset.promptEn : button.dataset.promptFr;
  resizeInput();
  elements.input.focus();
}));
document.querySelector('[data-welcome-action="documents"]')?.addEventListener("click", () => openDocuments());
document.querySelector('[data-welcome-action="continue"]')?.addEventListener("click", async () => {
  if (state.conversations.length) await selectConversation(state.conversations[0].id);
  else elements.input.focus();
});
elements.input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.composer.requestSubmit(); } });
elements.composer.addEventListener("submit", (event) => { event.preventDefault(); if (state.streaming) state.abortController?.abort(); else submitQuestion(elements.input.value); });
elements.thinkingToggle.addEventListener("click", () => applyThinkingMode(!state.thinkingEnabled, true));
elements.documentsButton.addEventListener("click", () => openDocuments()); elements.attachButton.addEventListener("click", () => openDocuments({ chooseFile: true }));
elements.toolSessionsButton.addEventListener("click", openSessions);
elements.documentsClose.addEventListener("click", () => elements.documentsDialog.close());
elements.pdfInput.addEventListener("change", () => selectPdf(elements.pdfInput.files[0]));
elements.dropZone.addEventListener("dragover", (event) => { event.preventDefault(); elements.dropZone.classList.add("dragging"); });
elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("dragging"));
elements.dropZone.addEventListener("drop", (event) => { event.preventDefault(); elements.dropZone.classList.remove("dragging"); selectPdf(event.dataTransfer.files[0]); });
elements.uploadButton.addEventListener("click", uploadDocument);
elements.metricsButton.addEventListener("click", openMetrics); elements.metricsClose.addEventListener("click", () => elements.metricsDialog.close());
elements.settingsButton.addEventListener("click", () => { closeSidebar(); elements.settingsDialog.showModal(); }); elements.settingsClose.addEventListener("click", () => elements.settingsDialog.close());
elements.settingsDocumentsButton.addEventListener("click", () => { elements.settingsDialog.close(); openDocuments(); });
elements.accountButton.addEventListener("click", () => { closeSidebar(); elements.settingsDialog.showModal(); });
elements.logoutButton.addEventListener("click", logout);
elements.sessionsButton.addEventListener("click", openSessions); elements.sessionsClose.addEventListener("click", () => elements.sessionsDialog.close());
elements.sessionsList.addEventListener("click", async (event) => { const button = event.target.closest("[data-revoke-session]"); if (!button) return; try { await api(`/api/auth/sessions/${button.dataset.revokeSession}`, { method: "DELETE" }); await openSessions(); } catch (error) { toast(error.message); } });
elements.adminButton.addEventListener("click", () => openAdmin()); elements.adminClose.addEventListener("click", closeAdmin); elements.adminRefresh.addEventListener("click", () => loadAdmin());
document.querySelectorAll("[data-admin-tab]").forEach((button) => button.addEventListener("click", () => selectAdminTab(button.dataset.adminTab)));
document.querySelectorAll("[data-open-admin]").forEach((button) => button.addEventListener("click", () => openAdmin(button.dataset.openAdmin)));
document.querySelectorAll("[data-settings-admin]").forEach((button) => button.addEventListener("click", () => { elements.settingsDialog.close(); openAdmin(button.dataset.settingsAdmin); }));
elements.operationGrid.addEventListener("click", (event) => { const button = event.target.closest("[data-job-kind]"); if (button) launchAdminJob(button.dataset.jobKind); });
elements.operationForm.addEventListener("submit", submitConfiguredOperation);
elements.operationClose.addEventListener("click", closeOperationConfigurator);
elements.operationCancel.addEventListener("click", closeOperationConfigurator);
elements.facebookCollectorClose.addEventListener("click", closeFacebookCollector);
elements.facebookCancel.addEventListener("click", closeFacebookCollector);
elements.facebookCollectionForm.addEventListener("submit", submitFacebookCollection);
elements.facebookPageOptions.addEventListener("change", updateFacebookEstimate);
elements.facebookMaxPosts.addEventListener("input", updateFacebookEstimate);
elements.facebookSelectAll.addEventListener("click", () => { elements.facebookPageOptions.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = true; }); updateFacebookEstimate(); });
elements.facebookSelectNone.addEventListener("click", () => { elements.facebookPageOptions.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = false; }); updateFacebookEstimate(); });
elements.scheduleKind.addEventListener("change", updateScheduleFields);
elements.scheduleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const facebookScheduled = ["collect_facebook", "full_refresh"].includes(elements.scheduleKind.value);
  const pageKeys = [...elements.scheduleFacebookPages.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
  const payload = facebookScheduled ? { page_keys: pageKeys, max_posts_per_page: Number(elements.scheduleFacebookMaxPosts.value) } : {};
  if (facebookScheduled && !pageKeys.length) { toast("Sélectionnez au moins une page Facebook pour cette planification."); return; }
  if (facebookScheduled && !state.adminData?.facebook?.token_configured) { toast("Configurez d’abord le jeton Apify sur le serveur depuis la collecte Facebook."); return; }
  try {
    await api("/api/admin/schedules", { method: "POST", body: JSON.stringify({ name: elements.scheduleName.value.trim(), task_kind: elements.scheduleKind.value, interval_minutes: Number(elements.scheduleInterval.value), enabled: true, payload }) });
    elements.scheduleForm.reset(); elements.scheduleInterval.value = "10080"; updateScheduleFields(); await loadAdmin();
  } catch (error) { toast(error.message); }
});
elements.scheduleList.addEventListener("change", async (event) => { const input = event.target.closest("[data-toggle-schedule]"); if (!input) return; const row = input.closest("[data-schedule-id]"); try { await api(`/api/admin/schedules/${row.dataset.scheduleId}`, { method: "PATCH", body: JSON.stringify({ enabled: input.checked }) }); await loadAdmin(); } catch (error) { toast(error.message); } });
elements.scheduleList.addEventListener("click", async (event) => { const button = event.target.closest("[data-delete-schedule]"); if (!button) return; const row = button.closest("[data-schedule-id]"); if (!window.confirm("Supprimer cette planification ?")) return; try { await api(`/api/admin/schedules/${row.dataset.scheduleId}`, { method: "DELETE" }); await loadAdmin(); } catch (error) { toast(error.message); } });
elements.modelForm.addEventListener("submit", async (event) => { event.preventDefault(); try { const result = await api("/api/admin/settings", { method: "PATCH", body: JSON.stringify({ generation_model: elements.generationModel.value, performance_profile: elements.performanceProfile.value }) }); toast("Configuration enregistrée. Le pipeline sera rechargé au prochain message."); await loadAdmin(); await updateHealth(); } catch (error) { toast(error.message); } });
elements.userForm.addEventListener("submit", async (event) => { event.preventDefault(); try { await api("/api/admin/users", { method: "POST", body: JSON.stringify({ display_name: elements.newUserName.value.trim(), email: elements.newUserEmail.value.trim(), password: elements.newUserPassword.value, role: elements.newUserRole.value }) }); elements.userForm.reset(); await loadAdmin(); } catch (error) { toast(error.message); } });
elements.userList.addEventListener("change", async (event) => { const select = event.target.closest("[data-user-role]"); if (!select) return; const row = select.closest("[data-user-id]"); try { await api(`/api/admin/users/${row.dataset.userId}`, { method: "PATCH", body: JSON.stringify({ role: select.value }) }); await loadAdmin(); } catch (error) { toast(error.message); } });
elements.userList.addEventListener("click", async (event) => { const button = event.target.closest("[data-toggle-user]"); if (!button) return; const row = button.closest("[data-user-id]"); const user = state.adminData?.users?.find((item) => item.id === row.dataset.userId); if (!user) return; try { await api(`/api/admin/users/${user.id}`, { method: "PATCH", body: JSON.stringify({ is_active: !user.is_active }) }); await loadAdmin(); } catch (error) { toast(error.message); } });
elements.securitySessionList.addEventListener("click", async (event) => { const button = event.target.closest("[data-admin-revoke-session]"); if (!button) return; try { await api(`/api/admin/sessions/${button.dataset.adminRevokeSession}`, { method: "DELETE" }); toast("Session révoquée."); await loadAdmin(); } catch (error) { toast(error.message); } });
elements.authForm.addEventListener("submit", submitAuth);
elements.authRegister.addEventListener("click", () => { elements.authForm.reset(); showAuth("register"); });
elements.authForgot.addEventListener("click", () => showAuth("forgot"));
elements.authGuest.addEventListener("click", continueAsGuest);
elements.authBack.addEventListener("click", () => { elements.authForm.reset(); showAuth("login"); });
document.querySelectorAll("[data-theme-choice]").forEach((button) => button.addEventListener("click", () => applyTheme(button.dataset.themeChoice)));
document.querySelectorAll("[data-language-choice]").forEach((button) => button.addEventListener("click", () => applyLanguage(button.dataset.languageChoice)));
document.querySelectorAll("dialog:not(.settings-screen):not(.admin-screen)").forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); }));
elements.adminDialog.addEventListener("click", (event) => { if (event.target === elements.adminDialog) closeAdmin(); });

async function initializeWorkspace() {
  await loadConversations({ selectFirst: false }); startNewChat();
  await updateHealth(); warmPipeline();
  window.clearInterval(state.healthTimer); state.healthTimer = window.setInterval(updateHealth, 5000);
}

async function initialize() {
  applyTheme(localStorage.getItem("rag-ensea-theme-v2") || "dark");
  applyLanguage(state.language);
  applyThinkingMode(state.thinkingEnabled);
  resizeInput();
  try {
    const status = await api("/api/auth/status");
    state.authCapabilities = status;
    if (status.setup_required) { showAuth("setup"); return; }
    if (state.resetToken) { showAuth("reset"); return; }
    try { applyCurrentUser(await api("/api/auth/me")); await initializeWorkspace(); }
    catch (_error) { showAuth("login"); }
  } catch (error) { showAuth("login"); elements.authError.textContent = "Le serveur est indisponible."; elements.authError.hidden = false; }
}

initialize();
