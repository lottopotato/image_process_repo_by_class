
// IMAGE_process_part1Dlg.cpp: 구현 파일
//

#include "stdafx.h"
#include "IMAGE_process_part1.h"
#include "IMAGE_process_part1Dlg.h"
#include "afxdialogex.h"
#include "math.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CIMAGEprocesspart1Dlg 대화 상자



CIMAGEprocesspart1Dlg::CIMAGEprocesspart1Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_IMAGE_PROCESS_PART1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CIMAGEprocesspart1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CIMAGEprocesspart1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
END_MESSAGE_MAP()


// CIMAGEprocesspart1Dlg 메시지 처리기

BOOL CIMAGEprocesspart1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	ShowWindow(SW_MAXIMIZE);

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	int m_width = 512;
	int m_height = 512;
	int m_size = m_width * m_height;

	FILE *imageFile;
	unsigned char *src_image = new unsigned char[m_size];
	x_image.Create(m_width, m_height, 32);

	if (0 == fopen_s(&imageFile, "512_512.raw", "rb")) {
		fread(src_image, m_size, 1, imageFile);
		// 0. vanilla
		Basic_image(src_image, m_width, m_height);
		
		// Arithmetic
		// 1. add pixel
		//ArithmeticPixel(src_image, m_width, m_height, 50, _T("Add"));
		// 2. Sub pixel
		//ArithmeticPixel(src_image, m_width, m_height, 50, _T("Sub"));
		// 3. Mul pixel
		//ArithmeticPixel(src_image, m_width, m_height, 2.5, _T("Mul"));
		// 4. Div pixel
		//ArithmeticPixel(src_image, m_width, m_height, 1.3, _T("Div"));

		// logic
		// 1. and
		//LogicPixel(src_image, m_width, m_height, _T("And"));
		// 2. or
		//LogicPixel(src_image, m_width, m_height, _T("Or"));
		// 3. xor
		//LogicPixel(src_image, m_width, m_height, _T("Xor"));

		// gamma correction
		//GammaCorrection(src_image, m_width, m_height, 0.5);

		delete[] src_image;
		fclose(imageFile);
	}
	
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CIMAGEprocesspart1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CIMAGEprocesspart1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		//CDialogEx::OnPaint();
		CPaintDC dc(this);
		x_image.Draw(dc, 10, 10);
		if (y_image != 0) {
			y_image.Draw(dc, 540, 10);
			dc.TextOutW(540, 10, result);
		}
		
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CIMAGEprocesspart1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CIMAGEprocesspart1Dlg::Basic_image(unsigned char *src_image, int width, int height)
{
	// TODO: 여기에 구현 코드 추가.
	int image_size = width * height;
	unsigned char *dest_image = _toPixel(src_image, image_size);
	::SetBitmapBits(x_image, image_size * 4, dest_image);
	delete[] dest_image;
}


unsigned char CIMAGEprocesspart1Dlg::clamping(int pixel)
{
	// TODO: 여기에 구현 코드 추가.
	if (pixel > 255){
		pixel = 255;
	}
	else if (pixel < 0) {
		pixel = 0;
	}
	return pixel;
}


int CIMAGEprocesspart1Dlg::ArithmeticPixel(unsigned char *src_image, int width, int height,
	int meticPixel, CString mode)
{
	// TODO: 여기에 구현 코드 추가.
	result = mode;
	unsigned char pixel = 0;
	int image_size = width * height;
	
	for (int i = 0; i < image_size; i++) {
		pixel = src_image[i];
		if (mode == "Add")
			src_image[i] = clamping(pixel + meticPixel);
		else if (mode == "Sub")
			src_image[i] = clamping(pixel - meticPixel);
		else if (mode == "Mul")
			src_image[i] = clamping(pixel * meticPixel);
		else if (mode == "Div")
			src_image[i] = clamping(pixel / meticPixel);
		else
			return 0;
	}
	unsigned char *dest_image = _toPixel(src_image, image_size);
	y_image.Create(width, height, 32);
	::SetBitmapBits(y_image, image_size * 4, dest_image);
	delete[] dest_image;
	return 0;
}

int CIMAGEprocesspart1Dlg::LogicPixel(unsigned char *src_image, int width, int height,
	CString mode)
{
	// TODO: 여기에 구현 코드 추가.
	result = mode;
	int image_size = width * height;
	// mask image
	unsigned char *mask_image = new unsigned char[image_size];
	for (int i = 0; i < image_size; i++) {
		if (i > 10000 && i < 100000)
			mask_image[i] = 255;
		else
			mask_image[i] = 0;
	}
	for (int i= 0; i < image_size; i++) {
		if (mode == "And") {
			src_image[i] = clamping(src_image[i] & mask_image[i]);
		}
		else if (mode == "Or") {
			src_image[i] = clamping(src_image[i] | mask_image[i]);
		}
		else if (mode == "Xor") {
			src_image[i] = clamping(src_image[i] ^ mask_image[i]);
		}
		else
			return 0;
	}
	unsigned char *dest_image = _toPixel(src_image, image_size);
	y_image.Create(width, height, 32);
	::SetBitmapBits(y_image, image_size * 4, dest_image);
	delete[] dest_image;
	
	return 0;
}

int CIMAGEprocesspart1Dlg::GammaCorrection(unsigned char *src_image, int width, int height,
	double gamma_val)
{
	result = _T("Gamma Correction");
	int image_size = width * height;
	for (int i = 0; i < image_size; i++) {
		// gamma output = 256 * pow ([input/256], r)
		src_image[i] = clamping(256 * pow((float)src_image[i] / 256, gamma_val));
	}
	unsigned char *dest_image = _toPixel(src_image, image_size);
	y_image.Create(width, height, 32);
	::SetBitmapBits(y_image, image_size * 4, dest_image);
	delete[] dest_image;

	return 0;
}


unsigned char* CIMAGEprocesspart1Dlg::_toPixel(unsigned char* src_image, int image_size)
{
	// TODO: 여기에 구현 코드 추가.
	unsigned char *dest_image = new unsigned char[image_size * 4];

	unsigned char *src_pos = src_image;
	unsigned char *dest_pos = dest_image;
	
	unsigned char pixel = 0;
	for (int i = 0; i < image_size; i++) {
		pixel = *src_pos;
		for (int j = 0; j < 3; j++) {
			*dest_pos++ = pixel;
		}
		*dest_pos++ = 0xFF;
		*src_pos++;
	}
	return dest_image;
}
