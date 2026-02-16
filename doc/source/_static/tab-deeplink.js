// Update the URL bar when a sphinx-design tab is clicked,
// so the link can be copied and shared to land on a specific tab.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".sd-tab-label").forEach(function (label) {
    label.addEventListener("click", function () {
      var syncId = this.getAttribute("data-sync-id");
      var syncGroup = this.getAttribute("data-sync-group");
      if (!syncId || !syncGroup) return;
      var params = new URLSearchParams(window.location.search);
      params.set(syncGroup, syncId);
      history.replaceState(null, "", "?" + params.toString() + window.location.hash);
    });
  });
});
